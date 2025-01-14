_base_ = [
    "../_base_/models/face_seg.py",
    "../_base_/datasets/loveda.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/cloud_scedule.py",
]

data_preprocessor = dict(size=(512, 512))
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(model="swinv2_base", input_resolution=512,feature_fuse=dict(type='FeatureFuse',norm="LN")),
    decode_head=dict(
        type="OursDecoder",
        token_lens=7,
        transformer=dict(type="OursTwoWayTransformer", depth=1),
        num_classes=7, 
    ),
)
