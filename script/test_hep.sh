python src/main.py  --model_name "/data/model/deepseek-coder-1.3b-instruct" \
                    --task "HumanEvalPlus" \
                    --batch_size 164 \
                    --prompt_type "Completion" \
                    --model_type "Chat" \
                    --prompt_prefix $'Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\n' \
                    --prompt_suffix $'\n```\n' \
                    --response_prefix $'' \
                    --response_suffix $'\n```\n' \
                    --trust_remote_code