"""Exercise original aggregation and the seed-aware HZ-swap configuration."""

from copy import deepcopy
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from mmengine.config import Config
from mmengine.logging import MessageHub
from mmengine.registry import HOOKS
from mmdet3d.registry import MODELS
from mmdet3d.utils import register_all_modules

import oneformer3d
from oneformer3d.query_decoder import MambaAggregation
from oneformer3d.seed_aware_query import SeedAwareQuerySampler
from oneformer3d.structures import InstanceData_


class PrefixScan(nn.Module):
    def forward(self, x):
        self.last_shape = x.shape
        return x.cumsum(dim=1)


class OriginalMambaAggregationTest(unittest.TestCase):
    def test_parameter_layout_and_original_formula(self):
        torch.manual_seed(23)
        layer = MambaAggregation(d_model=4, k=3)
        self.assertEqual(set(layer.state_dict()), {
            'w_k', 'w_b', 'w_q.weight', 'w_v.weight', 'w_o.weight',
            'norm.weight', 'norm.bias'})
        for n_source in (2, 7):
            for spatial in (False, True):
                with self.subTest(n_source=n_source, spatial=spatial):
                    query = torch.randn(2, 4, requires_grad=True)
                    source = torch.randn(n_source, 4, requires_grad=True)
                    query_pos = torch.randn(2, 3) if spatial else None
                    source_pos = torch.randn(n_source, 3) if spatial else None
                    q, v = layer.w_q(query), layer.w_v(source)
                    if spatial:
                        distances = torch.cdist(query_pos, source_pos)
                    else:
                        distances = -F.normalize(q, dim=-1) @ F.normalize(v, dim=-1).T
                    k = min(layer.k, n_source)
                    indices = distances.topk(k, largest=False).indices
                    weights = F.linear(q, layer.w_k[:k], layer.w_b[:k]).softmax(-1)
                    expected = layer.norm(
                        query + layer.w_o((weights[..., None] * q[:, None] * v[indices]).sum(1)))
                    actual = layer(query, source, query_pos, source_pos)
                    torch.testing.assert_close(actual, expected)
                    (actual * torch.randn_like(actual)).sum().backward()
                    self.assertTrue(torch.isfinite(query.grad).all())
                    self.assertTrue(torch.isfinite(source.grad).all())


class SeedAwareHZSwapConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_all_modules()
        cls.base = Config.fromfile(
            'configs/WHU-Stree/forestmamba_whu_stree_seed_aware_hilbert.py')
        cls.config = Config.fromfile(
            'configs/WHU-Stree/forestmamba_whu_stree_seed_aware_hz_swap.py')

    def test_configuration_inherits_seed_aware_training_settings(self):
        expected = deepcopy(self.base.model)
        expected.decoder.serialization_order = 'hz-swap'
        self.assertEqual(self.config.model, expected)
        self.assertEqual(self.config.train_dataloader, self.base.train_dataloader)
        self.assertEqual(self.config.val_dataloader, self.base.val_dataloader)
        self.assertEqual(self.config.train_cfg, self.base.train_cfg)
        self.assertEqual(self.config.train_cfg.max_epochs, 3000)
        self.assertEqual(self.config.train_cfg.val_interval, 50)
        checkpoint = deepcopy(self.config.default_hooks.checkpoint)
        self.assertEqual(checkpoint.pop('save_best'), ['pq', 'all_ap'])
        self.assertEqual(checkpoint.pop('rule'), ['greater', 'greater'])
        self.assertTrue(checkpoint.pop('save_last'))
        self.assertEqual(checkpoint, self.base.default_hooks.checkpoint)
        self.assertEqual(self.config.model.decoder.hz_coarse_bits, 1)
        self.assertEqual(self.config.work_dir,
                         './work_dirs/forestmamba_whu_stree_seed_aware_hz_swap')

    def test_model_sampling_and_decoder_backward(self):
        model = MODELS.build(self.config.model)
        self.assertIsInstance(model.seed_query, SeedAwareQuerySampler)
        self.assertEqual(model.test_cfg.query_mode, 'seed_aware')
        for layer in model.decoder.ssm_layers:
            self.assertEqual(layer.serialization_order, 'hz-swap')
            layer.mamba = PrefixScan()

        features = torch.randn(12, 32, requires_grad=True)
        positions = torch.randn(12, 3)
        ids = torch.arange(12)
        seeds, labels = model._sample_queries(
            features, torch.randn(12, 5), positions, ids, ids, 3)
        self.assertEqual(seeds.shape, (3, 32))
        torch.testing.assert_close(seeds, features[labels])
        torch.testing.assert_close(model._query_xyzs_cache, positions[labels])
        result = model.decoder(
            [features], [seeds], query_xyzs=[model._query_xyzs_cache],
            src_xyzs=[positions])
        self.assertEqual(result['masks'][0].shape, (5, 12))
        for layer in model.decoder.ssm_layers:
            self.assertEqual(layer.mamba.last_shape, (1, 5, 256))
        result['masks'][0].square().mean().backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_configured_semantic_and_instance_losses_backpropagate(self):
        criterion = MODELS.build(self.config.model.criterion)
        masks = torch.randn(5, 6, requires_grad=True)
        scores = torch.rand(5, 1, requires_grad=True)
        aux_masks = torch.randn(5, 6, requires_grad=True)
        aux_scores = torch.rand(5, 1, requires_grad=True)
        pred = dict(
            masks=[masks], scores=[scores],
            aux_outputs=[dict(masks=[aux_masks], scores=[aux_scores])])
        ground_truth = InstanceData_(
            sp_sem_masks=torch.tensor([[1, 1, 0, 0, 0, 0],
                                       [0, 0, 1, 1, 1, 1]]),
            sp_inst_masks=torch.tensor([[0, 0, 1, 1, 0, 0],
                                        [0, 0, 0, 0, 1, 1]]),
            labels_3d=torch.tensor([0, 0]),
            ratio_inspoint=torch.tensor([1., 1.]),
            query_inslabel=torch.tensor([0, 1, -1]))
        losses = criterion(pred, [ground_truth])
        self.assertEqual(set(losses),
                         {'inst_loss', 'last_layer_seg_loss', 'layer_0_seg_loss'})
        sum(losses.values()).backward()
        for tensor in (masks, scores, aux_masks, aux_scores):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_checkpoint_hook_keeps_separate_best_metrics_and_last(self):
        with TemporaryDirectory() as work_dir:
            class RunnerStub:
                def __init__(self):
                    self.work_dir = work_dir
                    self.epoch = 0
                    self.max_epochs = 101
                    self.logger = logging.getLogger('checkpoint-test')
                    self.message_hub = MessageHub.get_instance(
                        f'checkpoint-test-{id(self)}')

                def save_checkpoint(self, out_dir, filename, *args, **kwargs):
                    Path(out_dir, filename).write_text('checkpoint')

            runner = RunnerStub()
            hook = HOOKS.build(self.config.default_hooks.checkpoint)
            hook.before_train(runner)

            for epoch, metrics in ((50, {'pq': 0.70, 'all_ap': 0.80}),
                                   (100, {'pq': 0.80, 'all_ap': 0.70})):
                runner.epoch = epoch - 1
                hook.after_train_epoch(runner)
                runner.epoch = epoch
                hook.after_val_epoch(runner, metrics)

            runner.epoch = 100
            hook.after_train_epoch(runner)
            directory = Path(work_dir)
            self.assertTrue((directory / 'best_pq_epoch_100.pth').is_file())
            self.assertTrue((directory / 'best_all_ap_epoch_50.pth').is_file())
            self.assertFalse((directory / 'best_pq_epoch_50.pth').exists())
            self.assertTrue((directory / 'epoch_101.pth').is_file())
            self.assertEqual(Path((directory / 'last_checkpoint').read_text()).name,
                             'epoch_101.pth')

if __name__ == '__main__':
    unittest.main()
