_base_ = [
    "../_base_/models/face_seg.py",
    "../_base_/datasets/loveda.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/cloud_scedule.py",
]

data_preprocessor = dict(size=(512, 512))
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(model="swinv2_base", input_resolution=512,feature_fuse=dict(type='FeatureFuse',norm="LN"),frozen_backbone=True),
    decode_head=dict(
        type="OursDecoder",
        token_lens=7,
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
    dict(type="LinearLR", start_factor=1e-3, by_epoch=False, begin=0, end=320*5),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=0.9,
        begin=320*5,
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

batch_size=4
train_dataloader=dict(batch_size=batch_size)
val_dataloader=dict(batch_size=batch_size)
test_dataloader=dict(batch_size=batch_size)