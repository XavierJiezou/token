# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
)

model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type="SegFaceCeleb",
        input_resolution=256,
        model="convnext_base",
    ),
    decode_head=dict(
        type="FaceDecoder",
        transformer=dict(
            type="TwoWayTransformer",
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
        in_channels=2048,
        channels=5,
        num_classes=5,
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.5),
            dict(type="DiceLoss", loss_weight=0.5),
        ],
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
