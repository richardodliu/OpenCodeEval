#!/usr/bin/env bash

# set -x

EVAL_DIR=$(dirname "$(readlink -f "$0")")
cd "$EVAL_DIR" || exit

echo "OpenCodeEval/eval: $EVAL_DIR"

export MKL_THREADING_LAYER=GNU
export WEBHOOK_URL=''

CKPT_PATH=
CONFIG_PATH=config/benchmark.json
INTERVAL=60 # 监控文件夹是否有新ckpt产出的间隔

source ../ vllm
python main.py  --checkpoint_path $CKPT_PATH \
                --config_path $CONFIG_PATH \
                --interval $INTERVAL \
                --feishu_msg