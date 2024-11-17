cd /data/oce/oce_new

python src/main.py  --model_name "/data/model/deepseek-coder-1.3b-instruct" \
                --task "LeetCode" \
                --batch_size 180 \
                --max_tokens 4096 \
                --prompt_type "Instruction" \
                --prompt_prefix "" \
                --prompt_suffix "You need first to write a step-by-step outline and then write the code." \
                --trust_remote_code