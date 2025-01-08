_base_ = [
    "../_base_/models/face_seg.py",
    "../_base_/datasets/grass.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/grass_schedule.py",
]

data_preprocessor = dict(size=(256, 256))
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(model="swin_base",out_chans=64),
    decode_head=dict(
        type="OursDecoder",
        token_lens=10,
        transformer=dict(type="OursTwoWayTransformer", depth=1,embedding_dim=64),
        transformer_dim=64,
    ),
)
