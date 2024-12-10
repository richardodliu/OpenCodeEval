python src/main.py  --model_name "/data/model/deepseek-coder-1.3b-instruct" \
                    --task "MBPPPlus" \
                    --batch_size 378 \
                    --max_tokens 2048 \
                    --prompt_type "Instruction" \
                    --prompt_prefix $'Please provide a self-contained Python script that solves the following problem in a markdown code block:\n' \
                    --prompt_suffix $'' \
                    --response_prefix $'' \
                    --response_suffix $'\n```\n' \
                    --trust_remote_code