#!/usr/bin/env bash

# set -x

EVAL_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$EVAL_DIR")
cd "$PARENT_DIR" || exit

echo "OpenCodeEval: $PARENT_DIR"

export CUDA_VISIBLE_DEVICES=3

python src/main.py  --model_name "../models/deepseek-coder-1.3b-instruct" \
                    --task "MBPPPlus" \
                    --batch_size 378 \
                    --prompt_type "Instruction" \
                    --model_type "Chat" \
                    --prompt_prefix "" \
                    --prompt_suffix "" \
                    --trust_remote_code