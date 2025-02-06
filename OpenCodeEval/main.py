import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from OpenCodeEval.args import get_args, check_args
from OpenCodeEval.utils import refine_text, write_jsonl, stream_jsonl, calculate_pass_at_k

from OpenCodeEval.factory import BenchmarkFactory, BackendFactory

from tqdm.contrib.concurrent import process_map

def main():
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    args = check_args(args)

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    task = BenchmarkFactory.get_task(args)

    # get prompts
    prompts = task.get_prompt()

    for prompt in prompts:
        prompt['prompt'] = refine_text(args.prompt_prefix + prompt['prompt'] + args.prompt_suffix)
    prompts = sorted(prompts, key = lambda x: x['task_id'])
    write_jsonl(save_path + "/prompts.jsonl", prompts)

    # get generations
    # stop_words = task.chat_stop + task.base_stop if args.model_type == "Base" else task.chat_stop
    
    decoder = BackendFactory.get_backend(args)
    if decoder.is_chat():
        decoder.set_stop(task.chat_stop + task.base_stop)
    else:
        decoder.set_stop(task.base_stop)

    generations = decoder.generate(
        prompts,
        args.response_prefix,
        args.response_suffix
    )
    generations = sorted(generations, key = lambda x: (x['task_id'], x['completion_id']))
    write_jsonl(save_path + "/generations.jsonl", generations)

    # post-process generations
    generations = list(stream_jsonl(save_path + "/generations.jsonl"))
    solutions = process_map(
        task.postprocess_generation,
        generations,
        max_workers = args.num_workers,
        desc = "Post-processing Generations"
    )
    solutions = sorted(solutions, key = lambda x: (x['task_id'], x['completion_id']))
    write_jsonl(save_path + "/solutions.jsonl", solutions)

    # evaluate solutions
    solutions = list(stream_jsonl(save_path + "/solutions.jsonl"))
    evaluations = process_map(
        task.process_results,
        solutions,
        max_workers = args.num_workers,
        desc = "Evaluating Solutions"
    )
    evaluations = sorted(evaluations, key = lambda x: (x['task_id'], x['completion_id']))
    write_jsonl(save_path + "/evaluations.jsonl", evaluations)

    # calculate pass@k
    evaluations = list(stream_jsonl(save_path + "/evaluations.jsonl"))
    results = calculate_pass_at_k(evaluations, args.num_samples, args.list_k)
    write_jsonl(save_path + "/results.jsonl", results)


if __name__ == "__main__":
    main()