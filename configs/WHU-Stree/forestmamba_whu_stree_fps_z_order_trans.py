_base_ = ['./forestmamba_whu_stree.py']

# Scan Morton(X, Y, Z) and Morton(Y, X, Z) with shared SSM weights,
# restore both outputs to query order, then average.
model = dict(decoder=dict(serialization_order='z-order-trans'))

work_dir = './work_dirs/forestmamba_whu_stree_fps_z_order_trans'
