#!/bin/bash
# 并发评测脚本示例
# 使用方法: bash parallel_eval_example.sh

cd /volume/pt-train/users/rbliu/github/OpenCodeEval/agent
source /volume/pt-train/users/rbliu/miniconda3/bin/activate openrlhf

# 配置参数
CHECKPOINT_DIR="/volume/pt-train/users/rbliu/rbliu/jd/checkpoint/Qwen2.5-Coder-3B-Instruct/jd_dataset/epoch-1_batch-256_lr-1e-6"  # 修改为你的checkpoint目录
CONFIG_PATH="config/humaneval.json"           # benchmark配置文件
NUM_GPUS=1                                   # 每个checkpoint使用的GPU数
NUM_WORKERS=8                                # 每个checkpoint使用的CPU worker数
MAX_PARALLEL=8                               # 最大并行任务数（可选）

# 示例1: 基本用法（自动计算并行数）
python parallel_eval.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config_path "${CONFIG_PATH}" \
    --num_gpus ${NUM_GPUS} \
    --num_workers ${NUM_WORKERS} \
    2>&1 | tee "/volume/pt-train/users/rbliu/jd/logs/parallel_eval.log"

# 示例2: 指定最大并行数
# python parallel_eval.py \
#     --checkpoint_dir "${CHECKPOINT_DIR}" \
#     --config_path "${CONFIG_PATH}" \
#     --num_gpus ${NUM_GPUS} \
#     --num_workers ${NUM_WORKERS} \
#     --max_parallel ${MAX_PARALLEL}

# 示例3: 发送飞书通知（需要设置WEBHOOK_URL环境变量）
# export WEBHOOK_URL="your_webhook_url_here"
# python parallel_eval.py \
#     --checkpoint_dir "${CHECKPOINT_DIR}" \
#     --config_path "${CONFIG_PATH}" \
#     --num_gpus ${NUM_GPUS} \
#     --num_workers ${NUM_WORKERS} \
#     --feishu_msg
