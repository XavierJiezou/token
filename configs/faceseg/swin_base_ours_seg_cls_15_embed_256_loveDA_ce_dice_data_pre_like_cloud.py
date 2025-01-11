_base_ = [
    "../_base_/models/face_seg.py",
    # "../_base_/datasets/loveda.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/cloud_scedule.py",
]

dataset_type = 'LoveDADataset'
data_root = 'data/loveDA'

crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/train',
            seg_map_path='ann_dir/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))
# test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=["mIoU", "mDice", "mFscore"],)
test_evaluator = val_evaluator


data_preprocessor = dict(size=(512, 512))
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(model="swin_base", input_resolution=512),
    decode_head=dict(
        type="OursDecoder",
        token_lens=7,
        transformer=dict(type="OursTwoWayTransformer", depth=1),
        num_classes=7, 
    ),
)
