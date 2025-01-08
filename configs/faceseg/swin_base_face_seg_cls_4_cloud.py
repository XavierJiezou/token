_base_ = [
    "../_base_/models/face_seg.py",
    "../_base_/datasets/l8_biome.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/other_dataset_scedule.py",
]

data_preprocessor = dict(size=(512, 512))
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(model="swin_base",input_resolution=512),
    decode_head=dict(token_lens=4)
)
