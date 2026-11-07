_base_ = ['./forestmamba_whu_stree.py']

# ForestFormer3D: plain sparse U-Net, embedding-FPS query seeds, and a
# Transformer query decoder. Reuse the shared FPS model's padded sparse
# bounds and sampled validation on the eight labeled test scenes.
model = dict(
    decoder=dict(
        _delete_=True,
        type='ForAINetv2QueryDecoder_XAwarequery',
        num_layers=6,
        num_classes=1,
        num_instance_queries=0,
        num_semantic_queries=2,
        num_instance_classes=2,
        in_channels=32,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=True))

work_dir = './work_dirs/forestformer3d_whu_stree'
