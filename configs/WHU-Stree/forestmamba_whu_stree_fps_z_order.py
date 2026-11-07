_base_ = ['./forestmamba_whu_stree.py']

# Keep both shared-weight SSM streams in the same forward order, as in the
# Hilbert-identical experiment, using true 3-D Morton bit-interleaving instead.
model = dict(decoder=dict(serialization_order='z-order'))

work_dir = './work_dirs/forestmamba_whu_stree_fps_z_order'
