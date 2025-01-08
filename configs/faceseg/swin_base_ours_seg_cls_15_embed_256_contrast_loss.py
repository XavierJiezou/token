_base_ = [
    "../_base_/models/face_seg.py",
    "../_base_/datasets/grass.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/grass_schedule.py",
]

data_preprocessor = dict(size=(256, 256))
model = dict(
    type="ContrastLossEncoderDecoder",
    has_contrast_loss=True,
    lambda_contrast=100,
    data_preprocessor=data_preprocessor,
    backbone=dict(model="swin_base"),
    decode_head=dict(
        type="OursDecoder",
        token_lens=15,
        transformer=dict(type="OursTwoWayTransformer", depth=1),
    ),
)
