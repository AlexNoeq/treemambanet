_base_ = ['./forestmamba_whu_stree_seed_aware_hilbert.py']

# Keep the seed-aware sampler and training schedule; scan a single HZ order
# after swapping X/Y, with 0.2 m quantization and one coarse bit per axis.
model = dict(decoder=dict(serialization_order='hz-swap', hz_coarse_bits=1))

# Keep periodic and final checkpoints, plus separate best PQ and AP models.
default_hooks = dict(checkpoint=dict(
    save_best=['pq', 'all_ap'], rule=['greater', 'greater'], save_last=True))

work_dir = './work_dirs/forestmamba_whu_stree_seed_aware_hz_swap'
