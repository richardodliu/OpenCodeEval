#!/usr/bin/env bash

# set -x

EVAL_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$EVAL_DIR")
cd "$PARENT_DIR" || exit

echo "OpenCodeEval: $PARENT_DIR"

python src/main.py  --model_name "../models/deepseek-coder-1.3b-instruct" \
                    --task "HumanEval" \
                    --batch_size 164 \
                    --prompt_type "Completion" \
                    --model_type "Chat" \
                    --prompt_prefix $'Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\n' \
                    --prompt_suffix $'\n```\n' \
                    --trust_remote_code