#!/usr/bin/env bash

# set -x

EVAL_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$EVAL_DIR")
cd "$PARENT_DIR" || exit

echo "OpenCodeEval: $PARENT_DIR"

# 初始化 Conda（确保 Conda 的命令可用）
source /home/ma-user/work/code_dev/liurb/miniconda3/etc/profile.d/conda.sh

# 激活指定的环境
conda activate /home/ma-user/work/code_dev/liurb/miniconda3/envs/vllm

# 打印 Python 路径以验证
which python

python src/main.py  --task BirdDev \
                    --backend vllm \
                    --model_name $1 \
                    --save_path $2 \
                    --num_gpus 4 \
                    --max_tokens 8192 \
                    --temperature 0.0 \
                    --model_type Chat \
                    --time_out 30 \
                    --num_samples 1 \
                    --batch_size 100 \
                    --num_workers 10 \
                    --prompt_suffix $'\nPlease output valid SQLite query that solves the qustion in a markdown code block.'
