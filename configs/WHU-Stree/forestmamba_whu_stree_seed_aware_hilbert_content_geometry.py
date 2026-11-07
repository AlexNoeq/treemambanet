_base_ = ['./forestmamba_whu_stree_seed_aware_hilbert.py']

# Keep the existing 8-neighbor membership and query/value modulation.
# Replace rank weights with softmax(content similarity + relative-XYZ MLP).
# Geometry inputs: (source_x-query_x, source_y-query_y, source_z-query_z,
# Euclidean distance), in the original XYZ units. Semantic tokens use content
# weights only. All six decoder layers use this mode in training/inference.
# Inherit seed-aware sampling, Hilbert scans, matcher, losses and warm-up.
model = dict(decoder=dict(
    k=8,
    aggregation_weight_mode='content_geometry',
    aggregation_geometry_hidden_dim=32))

work_dir = './work_dirs/forestmamba_whu_stree_seed_aware_hilbert_content_geometry'
