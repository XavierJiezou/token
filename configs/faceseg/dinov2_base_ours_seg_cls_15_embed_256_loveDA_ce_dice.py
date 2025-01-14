_base_ = [
    "../_base_/models/face_seg.py",
    "../_base_/datasets/loveda.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/cloud_scedule.py",
]

data_preprocessor = dict(size=(512, 512))
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(model="dinov2", input_resolution=512, dinov2_config=dict(
        type="mmpretrain.VisionTransformer",
        arch="base",
        frozen_stages=12,
        img_size=512,
        patch_size=14,
        layer_scale_init_value=1e-5,
        out_indices=(2, 5, 8, 11),
        out_type = 'featmap',
        init_cfg=dict(
            type="Pretrained",
            checkpoint="checkpoints/dinov2-base.pth",
            prefix="backbone",
        ),)),
    decode_head=dict(
        type="OursDecoder",
        token_lens=7,
        transformer=dict(type="OursTwoWayTransformer", depth=1),
        num_classes=7, 
    ),
)
