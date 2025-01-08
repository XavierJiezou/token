_base_ = [
    "../_base_/models/face_seg.py",
    "../_base_/datasets/grass.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/grass_schedule.py",
]

data_preprocessor = dict(size=(256, 256))
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(model="swin_base"),
    decode_head=dict(
        type="OursDecoder",
        token_lens=15,
        transformer=dict(type="OursTwoWayTransformer", depth=1),
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.2),
            dict(type="DiceLoss", loss_weight=1),
            dict(
                type="FocalLoss",
                use_sigmoid=True,
                loss_weight=0.8,
                class_weight=[
                    0.19091777719422895,
                    0.1780360677687002,
                    0.24845635554870918,
                    0.30570702455617793,
                    0.07688277493218372,
                ],
            ),
        ],
    ),
)
