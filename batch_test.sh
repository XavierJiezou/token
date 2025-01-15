#!/bin/bash

# 定义路径列表
paths=(
  "work_dirs/efficientnet_base_ours_seg_cls_15_embed_256_loveDA_ce_dice"
  "work_dirs/swin_base_ours_seg_cls_15_embed_256_loveDA_ce_dice"
  "work_dirs/swin_base_ours_seg_cls_15_embed_256_loveDA_ce_dice_data_pre_like_cloud"
  "work_dirs/swin_base_ours_seg_cls_15_embed_256_loveDA_ce_dice_token_14"
  "work_dirs/swin_base_ours_seg_cls_15_embed_256_loveDA_ce_dice_token_21"
  "work_dirs/swinv2_base_ours_seg_cls_15_embed_256_loveDA_ce_dice_1024"
)

# 定义保存结果的父目录
output_parent_dir="data/loveDA/test_dir"

# 设置GPU设备ID
gpu_device=0

# 定义存储路径和文件的数组
declare -A model_files

# 遍历路径列表并找到所有的best模型文件
echo "正在查找所有模型的最佳权重文件..."
for path in "${paths[@]}"; do
  # 提取路径的最后一个目录名作为文件名
  model_name=$(basename "$path")
  
  # 查找以best开头的模型文件
  best_model_file=$(find "$path" -type f -name "best*.pth" | head -n 1)

  if [ -z "$best_model_file" ]; then
    echo "未找到以 'best' 开头的模型文件：$path"
  else
    echo "找到的模型文件：$best_model_file"
    model_files["$path"]="$best_model_file"
  fi
done

# 检查是否找到任何模型文件
if [ ${#model_files[@]} -eq 0 ]; then
  echo "未找到任何模型文件，退出。"
  exit 1
fi

# 显示找到的模型文件并询问是否继续
echo "以下是找到的模型文件："
for path in "${!model_files[@]}"; do
  echo "模型路径：$path"
  echo "权重文件：${model_files[$path]}"
done

read -p "是否继续执行所有模型测试 (y/n)? " confirm
if [[ "$confirm" != "y" ]]; then
  echo "操作已取消。"
  exit 0
fi

# 批量执行命令
for path in "${!model_files[@]}"; do
  # 提取路径的最后一个目录名作为文件名
  model_name=$(basename "$path")

  # 定义weight_path和config_path
  weight_path="${model_files[$path]}"
  config_path="${path}/${model_name}.py"

  # 定义保存结果的目录
  save_dir="${output_parent_dir}/${model_name}"
  save_path="${save_dir}/Result"

  # 创建目录（如果不存在）
  mkdir -p "$save_path"

  # 执行命令
  CUDA_VISIBLE_DEVICES=$gpu_device python tools/online_test/loveDA_test.py \
    --weight_path "$weight_path" \
    --config_path "$config_path" \
    --batch_size 16 \
    --save_path "$save_path"

  # 压缩Result目录到对应路径
  zip_file_path="${save_dir}/Result.zip"

  # 切换到保存目录的父目录以避免嵌套
  cd "$save_dir" || { echo "无法切换到目录：$save_dir"; continue; }

  # 使用相对路径压缩Result目录的内容，而不包含整个路径
  zip -r "Result.zip" Result/*

  # 检查压缩是否成功
  if [ $? -eq 0 ]; then
    echo "压缩完成：$zip_file_path"
  else
    echo "压缩失败：$zip_file_path"
  fi

  # 返回原始目录
  cd - > /dev/null
done
