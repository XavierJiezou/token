_base_ = [
    "../_base_/models/face_seg.py",
    "../_base_/datasets/cloud.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/cloud_scedule.py",
]

data_preprocessor = dict(size=(512, 512))
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(model="swin_base",input_resolution=512),
    decode_head=dict(
        type="OursDecoder",
        token_lens=15,
        transformer=dict(type="OursTwoWayTransformer", depth=1),
        num_classes=4,
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0)
        ],
    ),
)
