python src/main.py  --model_name "//data/model/deepseek-coder-1.3b-instruct" \
                    --task "MBPP" \
                    --batch_size 500 \
                    --max_tokens 4096 \
                    --prompt_type "Instruction" \
                    --prompt_prefix $'Please refer the given examples and generate a python function for my problem.\nExamples are listed as follows:\n' \
                    --prompt_suffix "" \
                    --response_prefix $'' \
                    --response_suffix $'\n```\n' \
                    --trust_remote_code