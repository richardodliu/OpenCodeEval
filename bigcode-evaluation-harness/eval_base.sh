accelerate launch  main.py \
  --model /data/model/deepseek-coder-1.3b-base \
  --tasks humaneval \
  --max_length_generation 2048 \
  --do_sample False \
  --n_samples 1 \
  --batch_size 1 \
  --allow_code_execution \
  --save_generations