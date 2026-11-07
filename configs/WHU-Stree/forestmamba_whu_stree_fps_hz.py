_base_ = ['./forestmamba_whu_stree.py']

# One forward HZ traversal: coarse Morton plus parity-selected local curves.
# Use 0.2 m quantization and adaptive depth with one coarse bit per axis.
model = dict(decoder=dict(serialization_order='hz', hz_coarse_bits=1))

work_dir = './work_dirs/forestmamba_whu_stree_fps_hz'
