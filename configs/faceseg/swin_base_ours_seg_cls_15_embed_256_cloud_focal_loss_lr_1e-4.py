_base_ = [
    "../_base_/models/face_seg.py",
    "../_base_/datasets/cloud.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/cloud_scedule.py",
]

# 添加 optim_wrapper 以覆盖学习率
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=0.0001,  # 覆盖学习率
        betas=(0.9, 0.999),
        weight_decay=0.01
    ),
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

data_preprocessor = dict(size=(256, 256))
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(model="swin_base"),
    decode_head=dict(
        type="OursDecoder",
        token_lens=15,
        transformer=dict(type="OursTwoWayTransformer", depth=1),
        num_classes=4,
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.2),
            dict(type="DiceLoss", loss_weight=1),
            dict(
                type="FocalLoss",
                use_sigmoid=True,
                loss_weight=0.8,
                class_weight=[
                    0.033644312588579764,
                    0.8387605073374941,
                    0.08845450935408698,
                    0.03914067071983916,
                ],
            ),
        ],
    ),
)
