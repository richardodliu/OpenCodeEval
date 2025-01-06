import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

from args import get_args, check_args
from utils import refine_text, write_jsonl, stream_jsonl, group_and_count, estimate_pass_at_k, multi_process_function

from factory import BenchmarkFactory, BackendFactory

def main():
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    args = check_args(args)

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    task = BenchmarkFactory.get_task(args)

    prompts = task.get_prompt()

    for prompt in prompts:
        prompt['prompt'] = refine_text(args.prompt_prefix + prompt['prompt'] + args.prompt_suffix)
    write_jsonl(save_path + "/prompts.jsonl", prompts)

    stop_words = task.chat_stop + task.base_stop if args.model_type == "Base" else task.chat_stop
    
    decoder = BackendFactory.get_backend(args)
    generations = decoder.generate(prompts,
                                   stop_words,
                                   args.response_prefix,
                                   args.response_suffix)
    write_jsonl(save_path + "/generations.jsonl", generations)
    generations = list(stream_jsonl(save_path + "/generations.jsonl"))

    solutions = multi_process_function(function = task.postprocess_generation,
                                       parameters = generations,
                                       num_workers = args.num_workers,
                                       desc = "Post-processing solutions")
    write_jsonl(save_path + "/solutions.jsonl", solutions)

    evaluations = multi_process_function(function = task.process_results,
                                             parameters = solutions,
                                             num_workers = args.num_workers,
                                             desc = "Evaluating solutions")
    write_jsonl(save_path + "/evaluation.jsonl", evaluations)

    result_list = group_and_count(evaluations, group_key = 'task_id', count_key = 'passed')
    pass_rate = float(np.mean(estimate_pass_at_k(num_samples = args.num_samples, num_correct = result_list, k = args.k)))
    write_jsonl(save_path + "/result.json", [{"score": pass_rate}])
    print(f"Pass@{args.k}:", pass_rate)

if __name__ == "__main__":
    main()