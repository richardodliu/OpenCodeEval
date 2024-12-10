cd /data/oce/oce_new

python src/main.py  --model_name "/data/model/deepseek-coder-1.3b-base" \
                --task "BigCodeHard" \
                --save_path "save" \
                --batch_size 148 \
                --prompt_type "Completion" \
                --trust_remote_code