#!/bin/bash

# 定义可变参数
device=$1
config=$2
weight=$3
batch_size=$4

# 执行 Python 脚本
CUDA_VISIBLE_DEVICES=$device python tools/test.py $config $weight --cfg-options model.test_cfg.mode=slide model.test_cfg.crop_size="(512,512)" model.test_cfg.stride="(512,512)" test_dataloader.batch_size=$batch_size test_evaluator.output_dir=data/tmp/$(basename $config .py)

python tools/post_process.py data/tmp/$(basename $config .py)
