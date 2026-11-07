_base_ = ['./forestmamba_whu_stree.py']

# Both streams scan the same forward Hilbert permutation with shared SSM
# weights, then restore query order and average their outputs.
model = dict(decoder=dict(serialization_order='hilbert-bidirectional'))

work_dir = './work_dirs/forestmamba_whu_stree_fps_hilbert_bidirectional'
