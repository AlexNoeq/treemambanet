_base_ = ['./forestmamba_whu_stree_fps_hz_bidirectional.py']

# Keep tree voxels and embedding-space FPS as the main selection criterion.
# Add seedness weighting and a normalized shifted-center distance term.
# GT instance centroids supervise loss_seedness and loss_seed_offset from
# epoch zero; the inherited decoder warm-up still uses prepare_epoch=20.
model = dict(
    type='ForAINetV2OneFormer3D_SeedAware_Spatial',
    seed_query=dict(
        type='SeedAwareQuerySampler',
        hidden_channels=64,
        center_sigma=0.5,
        min_radius=0.2,
        seed_loss_weight=1.0,
        offset_loss_weight=1.0,
        embedding_weight=0.7,  # 1.0 uses only embedding distance, still score-guided
        score_power=2.0,
        score_floor=0.05,
        # Top-score cap bounds sampling cost; None considers all tree voxels.
        max_candidates=32768),
    test_cfg=dict(query_mode='seed_aware'))

work_dir = './work_dirs/forestmamba_whu_stree_seed_aware_hz_bidirectional'
