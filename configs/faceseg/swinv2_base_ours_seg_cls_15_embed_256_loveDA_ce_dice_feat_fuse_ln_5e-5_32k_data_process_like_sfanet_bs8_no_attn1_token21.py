_base_ = [
    "../_base_/models/face_seg.py",
    "../_base_/datasets/loveda.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/cloud_scedule.py",
]

data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    size=(512, 512),
)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        model="swinv2_base",
        input_resolution=512,
        feature_fuse=dict(type="FeatureFuse", norm="LN"),
    ),
    decode_head=dict(
        type="OursDecoder",
        token_lens=21,
        transformer=dict(type="OursTwoWayTransformer", depth=1,do_attn1=False),
        num_classes=7,
    ),
)

# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=5e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)
# learning policy
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-3, by_epoch=False, begin=0, end=320 * 5),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=0.9,
        begin=320 * 5,
        end=32000,
        by_epoch=False,
    ),
]
# training schedule for 40k
train_cfg = dict(type="IterBasedTrainLoop", max_iters=32000, val_interval=320)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=320, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=320,
        save_best=["mIoU"],
        rule=["greater"],
        max_keep_ckpts=1,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)

# dataset settings
dataset_type = "LoveDADataset"
data_root = "data/loveDA"
crop_size = (512, 512)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=True),
    dict(
        type="RandomResize",
        scale=crop_size,
        ratio_range=(0.75,1.5),
        keep_ratio=True,
    ),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type="RandomFlip", prob=[0.5, 0.5], direction=["horizontal", "vertical"]),
    dict(
        transforms=[
            dict(type="RandomRotate90",p=0.5),
            dict(type="HorizontalFlip",p=0.5),
            dict(type="VerticalFlip",p=0.5),
        ],
        type="Albu",
    ),
    # dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type="LoadAnnotations", reduce_zero_label=True),
    dict(type="PackSegInputs"),
]
val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations", reduce_zero_label=True),
    dict(type="PackSegInputs"),
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(
        type="TestTimeAug",
        transforms=[
            [dict(type="Resize", scale_factor=r, keep_ratio=True) for r in img_ratios],
            [
                dict(type="RandomFlip", prob=0.0, direction="horizontal"),
                dict(type="RandomFlip", prob=1.0, direction="horizontal"),
            ],
            # [dict(type="LoadAnnotations")],
            [dict(type="PackSegInputs")],
        ],
    ),
]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="img_dir/train", seg_map_path="ann_dir/train"),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="img_dir/val", seg_map_path="ann_dir/val"),
        pipeline=val_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="img_dir/test"),
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(
    type="IoUMetric",
    iou_metrics=["mIoU", "mDice", "mFscore"],
)
test_evaluator = dict(
    type="IoUMetric",
    iou_metrics=["mIoU", "mDice", "mFscore"],
    format_only=True,
    output_dir="data/loveDA/test_dir",
)
