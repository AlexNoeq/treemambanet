_base_ = ['./forestmamba_whu_stree.py']

# One HZ permutation scanned forward and backward with shared SSM weights.
# Use 0.2 m quantization and adaptive depth with one coarse bit per axis.
model = dict(decoder=dict(
    serialization_order='hz-bidirectional', hz_coarse_bits=1))

work_dir = './work_dirs/forestmamba_whu_stree_fps_hz_bidirectional'
