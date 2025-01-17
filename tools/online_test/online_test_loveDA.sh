#!/bin/bash

# 定义可变参数
device=$1
config=$2
weight=$3
batch_size=$4

# 执行 Python 脚本
CUDA_VISIBLE_DEVICES=$device python tools/test.py $config $weight --tta --cfg-options test_dataloader.batch_size=$batch_size test_evaluator.output_dir=data/loveDA/test_dir/$(basename $config .py)

python tools/post_process.py data/loveDA/test_dir/$(basename $config .py)
