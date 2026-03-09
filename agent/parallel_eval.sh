#!/bin/bash
# 并发评测脚本示例
# 使用方法: bash parallel_eval_example.sh

cd /volume/pt-train/users/rbliu/github/OpenCodeEval/agent
source /volume/pt-train/users/rbliu/miniconda3/bin/activate openrlhf

# 配置参数
CHECKPOINT_DIR="/volume/pt-train/users/rbliu/checkpoint/leetcode/Qwen2.5-32B/cosine_diversity"  # 修改为你的checkpoint目录
CONFIG_PATH="config/leetcode.json"           # benchmark配置文件
NUM_GPUS=1                                   # 每个checkpoint使用的GPU数
NUM_WORKERS=4                                # 每个checkpoint使用的CPU worker数
MAX_PARALLEL=8                               # 最大并行任务数（可选）

# 示例1: 基本用法（自动计算并行数）
python parallel_eval.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config_path "${CONFIG_PATH}" \
    --num_gpus ${NUM_GPUS} \
    --num_workers ${NUM_WORKERS} \
    2>&1 | tee "/volume/pt-train/users/rbliu/github/OpenCodeEval/agent/parallel_eval.log"

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
