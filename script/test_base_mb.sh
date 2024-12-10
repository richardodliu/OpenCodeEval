cd /data/oce/oce_new

python src/main.py  --model_name "/data/model/deepseek-coder-1.3b-base" \
                    --task "MBPP" \
                    --batch_size 500 \
                    --max_tokens 4096 \
                    --model_type "Base" \
                    --prompt_type "Instruction" \
                    --prompt_prefix "" \
                    --prompt_suffix "" \
                    --trust_remote_code