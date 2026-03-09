CKPT_PATH=$1

python  /volume/pt-train/users/rbliu/github/OpenCodeEval/agent/main.py  \
        --checkpoint_path ${CKPT_PATH} \
        --config_path /volume/pt-train/users/rbliu/github/OpenCodeEval/agent/config/leetcode.json \
        --num_gpus 1 \
        --num_workers 4 \
        2>&1 | tee "${CKPT_PATH}/evaluation.log"