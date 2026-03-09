source /volume/pt-train/users/rbliu/miniconda3/bin/activate openrlhf

OpenCodeEval    --model_name /volume/pt-train/users/rbliu/checkpoint/leetcode/Qwen2.5-Coder-7B/random_dataset/epoch-3_batch-128_lr-1e-6/checkpoint-50 \
                --save_path /volume/pt-train/users/rbliu/checkpoint/leetcode/Qwen2.5-Coder-7B/random_dataset/epoch-3_batch-128_lr-1e-6/eval/checkpoint-50/leetcode_test \
                --task LeetCode \
                --backend vllm \
                --split test \
                --batch_size 2280 \
                --temperature 1.0 \
                --num_samples 10 \
                --list_k 1,2,3,4,5,6,7,8,9,10 \
                --num_gpus 1 \
                --num_workers 4 \
                --max_tokens 1024 \
                --time_out 3 \
                --prompt_type Instruction \
                --model_type Chat \
                --prompt_prefix '' \
                --prompt_suffix '' \
                --trust_remote_code \
                2>&1 | tee "/volume/pt-train/users/rbliu/github/OpenCodeEval/agent/agent_output.log"