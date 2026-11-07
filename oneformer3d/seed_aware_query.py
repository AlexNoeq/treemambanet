"""Learned seedness and center offsets for tree-embedding FPS queries."""

import torch
from torch import nn
from torch.nn import functional as F
from mmdet3d.registry import MODELS
from torch_scatter import scatter_mean

from .oneformer3d import ForAINetV2OneFormer3D_FPS_Spatial


@torch.no_grad()
def score_guided_fps(embeddings, shifted_xyz, scores, num_queries,
                     embedding_weight=0.7, score_power=2.0, score_floor=0.05,
                     max_candidates=32768):
    """Sample unique tree indices using seedness-weighted joint distances.

    Both distance terms are normalized by their scene's total variance:
    d(i,j) = w * ||e_i-e_j||^2 / var(e)
           + (1-w) * ||p'_i-p'_j||^2 / var(p').
    Start at the highest seedness, then maximize
    (floor + (1-floor)*score_i)^power * min_j d(i,j).
    Embeddings remain the main metric by default. No N-by-N matrix is built.
    The optional top-score candidate cap bounds the O(K*N) sampling cost.
    """
    if not 0 <= embedding_weight <= 1:
        raise ValueError('embedding_weight must be in [0, 1]')
    if score_power <= 0 or not 0 < score_floor <= 1:
        raise ValueError('score_power must be positive; score_floor in (0, 1]')
    if num_queries < 0 or (max_candidates is not None and max_candidates < 1):
        raise ValueError('Invalid query count or candidate cap')
    n = embeddings.shape[0]
    if embeddings.ndim != 2 or shifted_xyz.shape != (n, 3) or scores.shape != (n,):
        raise ValueError('Expected embeddings (N,D), shifted_xyz (N,3), scores (N,)')
    k = min(int(num_queries), n)
    if k == 0:
        return torch.empty(0, dtype=torch.long, device=embeddings.device)

    scores = scores.detach().float().clamp(0, 1)
    candidates = torch.arange(n, device=embeddings.device)
    if max_candidates is not None and n > max(k, max_candidates):
        candidates = scores.topk(max(k, max_candidates), sorted=True).indices
    scores = scores[candidates]
    weights = (score_floor + (1 - score_floor) * scores).pow(score_power)

    def normalize(values):
        values = values.detach().float()
        values = values - values.mean(0, keepdim=True)
        return values / values.square().sum(-1).mean().clamp_min(1e-12).sqrt()

    embed = normalize(embeddings[candidates])
    centers = normalize(shifted_xyz[candidates])
    nearest = torch.full_like(scores, float('inf'))
    selected = torch.empty(k, dtype=torch.long, device=embeddings.device)
    used = torch.zeros_like(scores, dtype=torch.bool)
    farthest = scores.argmax().reshape(1)
    for step in range(k):
        selected[step:step + 1] = candidates[farthest]
        used[farthest] = True
        distances = (
            embedding_weight * (embed - embed[farthest]).square().sum(-1)
            + (1 - embedding_weight)
            * (centers - centers[farthest]).square().sum(-1))
        nearest = torch.minimum(nearest, distances)
        # Coincident centers/embeddings still select unique, high-score points.
        priority = weights * nearest.clamp_min(1e-12)
        priority.masked_fill_(used, -1)
        farthest = priority.argmax().reshape(1)
    return selected


@MODELS.register_module()
class SeedAwareQuerySampler(nn.Module):
    """Predict O_i and meter-valued center offsets from backbone features.

    Training targets use instance centroids in the augmented input region.
    Seedness is exp(-0.5 * (distance / (sigma * instance_RMS_radius))^2).
    Ground and voxels with no strict instance majority have seedness zero
    and no offset supervision. Instance IDs need not be contiguous; tree ID
    zero is valid. The heads train before the decoder warm-up finishes.
    """

    def __init__(self, in_channels, hidden_channels=64, center_sigma=0.5,
                 min_radius=0.2, seed_loss_weight=1.0, offset_loss_weight=1.0,
                 embedding_weight=0.7, score_power=2.0, score_floor=0.05,
                 max_candidates=32768):
        super().__init__()
        if center_sigma <= 0 or min_radius <= 0:
            raise ValueError('center_sigma and min_radius must be positive')
        if seed_loss_weight < 0 or offset_loss_weight < 0:
            raise ValueError('Loss weights must be nonnegative')
        if not 0 <= embedding_weight <= 1:
            raise ValueError('embedding_weight must be in [0, 1]')
        if score_power <= 0 or not 0 < score_floor <= 1:
            raise ValueError('score_power must be positive; score_floor in (0, 1]')
        if max_candidates is not None and max_candidates < 1:
            raise ValueError('max_candidates must be positive or None')
        self.center_sigma = center_sigma
        self.min_radius = min_radius
        self.seed_loss_weight = seed_loss_weight
        self.offset_loss_weight = offset_loss_weight
        self.sampling_cfg = dict(
            embedding_weight=embedding_weight, score_power=score_power,
            score_floor=score_floor, max_candidates=max_candidates)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels), nn.ReLU(inplace=True))
        self.seedness_head = nn.Linear(hidden_channels, 1)
        self.offset_head = nn.Linear(hidden_channels, 3)
        # Begin with unshifted coordinates, then learn center displacement.
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

    def forward(self, features):
        hidden = self.shared_mlp(features)
        return self.seedness_head(hidden).squeeze(-1), self.offset_head(hidden)

    @torch.no_grad()
    def build_targets(self, point_xyz, point_to_voxel, instance_ids,
                      tree_mask, num_voxels):
        """Build voxel targets from points without a dense instance mask matrix."""
        xyz = point_xyz[:, :3].detach().float()
        # Local coordinates improve accumulation precision for georeferenced PLYs.
        xyz = xyz - xyz[:1]
        point_to_voxel = point_to_voxel.long()
        instance_ids = instance_ids.long()
        valid = tree_mask.bool() & (instance_ids >= 0)
        scores = xyz.new_zeros(num_voxels)
        offsets = xyz.new_zeros((num_voxels, 3))
        voxel_ids = instance_ids.new_full((num_voxels,), -1)
        targets = dict(seedness=scores, offsets=offsets, instance_ids=voxel_ids)
        if not valid.any():
            return targets

        unique_ids, compact_ids = torch.unique(instance_ids[valid], return_inverse=True)
        num_instances = unique_ids.numel()
        inst_counts = torch.bincount(compact_ids, minlength=num_instances).float()
        centers = xyz.new_zeros((num_instances, 3))
        centers.index_add_(0, compact_ids, xyz[valid])
        centers /= inst_counts[:, None]
        radii = xyz.new_zeros(num_instances)
        radii.index_add_(0, compact_ids,
                         (xyz[valid] - centers[compact_ids]).square().sum(-1))
        radii = (radii / inst_counts).sqrt().clamp_min(self.min_radius)

        counts = torch.bincount(point_to_voxel, minlength=num_voxels).float()
        voxel_xyz = xyz.new_zeros((num_voxels, 3))
        voxel_xyz.index_add_(0, point_to_voxel, xyz)
        voxel_xyz /= counts[:, None].clamp_min(1)
        point_ids = torch.zeros_like(instance_ids)
        point_ids[valid] = compact_ids + 1  # zero denotes background
        keys, votes = torch.unique(
            point_to_voxel * (num_instances + 1) + point_ids, return_counts=True)
        voxels = torch.div(keys, num_instances + 1, rounding_mode='floor')
        labels = keys.remainder(num_instances + 1) - 1
        majority = (labels >= 0) & (votes > counts[voxels] * 0.5)
        voxel_ids[voxels[majority]] = labels[majority]
        fg = voxel_ids >= 0
        offsets[fg] = centers[voxel_ids[fg]] - voxel_xyz[fg]
        relative_distance = offsets[fg].norm(dim=-1) / (
            self.center_sigma * radii[voxel_ids[fg]])
        scores[fg] = torch.exp(-0.5 * relative_distance.square())
        return targets

    @staticmethod
    def _instance_mean(values, ids):
        """Give each tree equal weight despite different point counts."""
        unique_ids, inverse = torch.unique(ids, return_inverse=True)
        sums = values.new_zeros(unique_ids.numel())
        sums.index_add_(0, inverse, values)
        counts = torch.bincount(inverse, minlength=sums.numel())
        return (sums / counts).mean()

    def loss(self, features, point_xyz, point_to_voxel, instance_ids, tree_mask):
        logits, offsets = self(features)
        targets = self.build_targets(
            point_xyz, point_to_voxel, instance_ids, tree_mask, features.shape[0])
        fg = targets['instance_ids'] >= 0
        seed_loss = logits.sum() * 0
        offset_loss = offsets.sum() * 0
        if logits.numel():
            point_loss = F.binary_cross_entropy_with_logits(
                logits.float(), targets['seedness'], reduction='none')
            groups = []
            if fg.any():
                groups.append(self._instance_mean(
                    point_loss[fg], targets['instance_ids'][fg]))
                regression = F.smooth_l1_loss(
                    offsets[fg].float(), targets['offsets'][fg], reduction='none')
                offset_loss = self._instance_mean(
                    regression.mean(-1), targets['instance_ids'][fg])
            if (~fg).any():
                groups.append(point_loss[~fg].mean())
            seed_loss = torch.stack(groups).mean()
        return dict(loss_seedness=seed_loss * self.seed_loss_weight,
                    loss_seed_offset=offset_loss * self.offset_loss_weight)

    @torch.no_grad()
    def sample(self, features, embeddings, xyz, num_queries):
        logits, offsets = self(features)
        # Apply offsets in a local frame for stable distance calculations.
        local_centers = xyz[:, :3].float() - xyz[:1, :3].float() + offsets.float()
        indices = score_guided_fps(
            embeddings, local_centers, logits.sigmoid(), num_queries,
            **self.sampling_cfg)
        query_xyz = xyz[indices, :3].float() + offsets[indices].float()
        return indices, query_xyz


@MODELS.register_module()
class ForAINetV2OneFormer3D_SeedAware_Spatial(ForAINetV2OneFormer3D_FPS_Spatial):
    """Tree-embedding FPS guided by learned seedness and center offsets.

    Query content comes from selected backbone voxels; query positions are
    their predicted centers. Train/validation/sliding-window paths share the
    same sampler. The inherited backbone, decoder and warm-up stay intact.
    """

    def __init__(self, seed_query=None, **kwargs):
        super().__init__(**kwargs)
        head_cfg = dict(type='SeedAwareQuerySampler',
                        in_channels=kwargs['num_channels'],
                        min_radius=self.voxel_size)
        head_cfg.update(seed_query or {})
        self.seed_query = MODELS.build(head_cfg)

    def _query_auxiliary_losses(self, voxel_features, batch_inputs_dict,
                                batch_data_samples, coordinates, inverse_mapping):
        """Train the new head on attached backbone features in every epoch."""
        losses = {}
        for i, features in enumerate(voxel_features):
            mapping = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
            mapping = torch.unique(mapping, return_inverse=True)[1]
            gt = batch_data_samples[i].gt_pts_seg
            tree_mask = torch.isin(
                gt.pts_semantic_mask,
                gt.pts_semantic_mask.new_tensor(self.thing_cls))
            sample_losses = self.seed_query.loss(
                features, batch_inputs_dict['points'][i], mapping,
                gt.pts_instance_mask, tree_mask)
            for name, value in sample_losses.items():
                losses[name] = losses.get(name, 0) + value / len(voxel_features)
        return losses

    def _sample_queries(self, backbone_feats, embed_feats_tree, xyz_voxels,
                        tree_indices, voxel_instance_labels, max_queries):
        self._n_chm_queries = 0
        if tree_indices.numel() == 0:
            n = min(max_queries, backbone_feats.shape[0])
            self._query_indices_cache = torch.arange(n, device=backbone_feats.device)
            return super()._sample_queries(
                backbone_feats, embed_feats_tree, xyz_voxels, tree_indices,
                voxel_instance_labels, max_queries)
        indices, query_xyz = self.seed_query.sample(
            backbone_feats[tree_indices], embed_feats_tree,
            xyz_voxels[tree_indices], max_queries)
        selected = tree_indices[indices]
        self._query_indices_cache = selected
        self._query_xyzs_cache = query_xyz
        labels = (voxel_instance_labels[selected]
                  if voxel_instance_labels is not None else None)
        return backbone_feats[selected], labels

    def _build_window_queries(self, x0, embed_logits, tree_indices,
                              inverse_mapping2, pc3, mode):
        """Also support the inherited optional bm1/bm2 inference entry point."""
        mapping = torch.unique(inverse_mapping2, return_inverse=True)[1]
        voxel_xyz = scatter_mean(pc3[:, :3].float(), mapping, dim=0)
        query_feats, _ = self._sample_queries(
            x0, embed_logits[tree_indices], voxel_xyz, tree_indices,
            None, self.query_point_num)
        return (query_feats, self._query_xyzs_cache, voxel_xyz,
                self._query_indices_cache)
