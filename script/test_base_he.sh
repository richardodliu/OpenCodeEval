cd /data/oce/oce_new

python src/main.py  --model_name "/data/model/deepseek-coder-1.3b-base" \
                    --task "HumanEvalPlus" \
                    --num_samples 10 \
                    --temperature 1.0 \
                    --batch_size 164 \
                    --prompt_type "Completion" \
                    --model_type "Base" \
                    --trust_remote_code