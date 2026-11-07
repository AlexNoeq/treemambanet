_base_ = ['./forestmamba_whu_stree_seed_aware_hz_bidirectional.py']

# Hilbert + Hilbert-trans (swap X/Y), using shared Mamba weights.
# Inherit the seed-aware tree-embedding sampler, losses and training settings.
model = dict(decoder=dict(serialization_order='hilbert'))

work_dir = './work_dirs/forestmamba_whu_stree_seed_aware_hilbert'
