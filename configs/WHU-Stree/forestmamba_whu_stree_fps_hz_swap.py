_base_ = ['./forestmamba_whu_stree.py']

# One forward HZ traversal after swapping X/Y for coarse and local encoding.
# Use 0.2 m quantization and adaptive depth with one coarse bit per axis.
model = dict(decoder=dict(serialization_order='hz-swap', hz_coarse_bits=1))

work_dir = './work_dirs/forestmamba_whu_stree_fps_hz_swap'
