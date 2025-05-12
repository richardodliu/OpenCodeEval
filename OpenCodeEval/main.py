import os
import argparse

from OpenCodeEval.args import get_args, check_args
from OpenCodeEval.utils import refine_text, write_jsonl, stream_jsonl, calculate_pass_at_k

from OpenCodeEval.factory import BenchmarkFactory, BackendFactory

from tqdm.contrib.concurrent import process_map, thread_map

def main():

    parser = argparse.ArgumentParser()
    args = get_args(parser)
    args = check_args(args)

    save_path = args.save_path
    os.makedirs(save_path, exist_ok = True)

    task = BenchmarkFactory.get_task(args)

    # get prompts
    prompts = task.get_prompt()

    for prompt in prompts:
        prompt['prompt'] = refine_text(args.prompt_prefix + prompt['prompt'] + args.prompt_suffix)
    prompts = sorted(prompts, key = lambda x: x['task_id'])
    write_jsonl(os.path.join(save_path, "prompts.jsonl"), prompts)

    if args.split  ==  'plus':
        base_path = save_path.replace('plus', 'base')
        if os.path.exists(os.path.join(base_path, "generations.jsonl")):
            generations = list(stream_jsonl(os.path.join(base_path, "generations.jsonl")))
            write_jsonl(os.path.join(save_path, "generations.jsonl"), generations)

    # check if generations exits
    if not os.path.exists(os.path.join(save_path, "generations.jsonl")):
        
        prompts = list(stream_jsonl(os.path.join(save_path, "prompts.jsonl")))
        # get generations
        decoder = BackendFactory.get_backend(args)
        if decoder.is_chat():
            decoder.set_stop(task.chat_stop)
        else:
            decoder.set_stop(task.chat_stop + task.base_stop)

        generations = decoder.generate(
            prompts,
            args.response_prefix,
            args.response_suffix
        )

        generations = sorted([data for data in generations if data['completion']], key = lambda x: (x['task_id'], x['completion_id']))
        write_jsonl(os.path.join(save_path, "generations.jsonl"), generations)

    else:
        
        generated_ids = [data['task_id'] for data in stream_jsonl(os.path.join(save_path, "generations.jsonl"))]
        prompts = [data for data in stream_jsonl(os.path.join(save_path, "prompts.jsonl")) if data['task_id'] not in generated_ids]
        if len(prompts) > 0:

            # get generations
            decoder = BackendFactory.get_backend(args)
            if decoder.is_chat():
                decoder.set_stop(task.chat_stop)
            else:
                decoder.set_stop(task.chat_stop + task.base_stop)

            continue_generations = decoder.generate(
                prompts,
                args.response_prefix,
                args.response_suffix
            )

            generations = sorted([data for data in continue_generations if data['completion']] + list(stream_jsonl(os.path.join(save_path, "generations.jsonl"))), key = lambda x: (x['task_id'], x['completion_id']))
            write_jsonl(os.path.join(save_path, "generations.jsonl"), generations)


    # post-process generations
    # if not os.path.exists(os.path.join(save_path, "solutions.jsonl")):
    if True:

        generations = list(stream_jsonl(os.path.join(save_path, "generations.jsonl")))
        solutions = thread_map(
            task.postprocess_generation,
            generations,
            max_workers = args.num_workers,
            desc = "Post-processing Generations"
        )
        solutions = sorted(solutions, key = lambda x: (x['task_id'], x['completion_id']))
        write_jsonl(os.path.join(save_path, "solutions.jsonl"), solutions)

    # evaluate solutions
    # if not os.path.exists(os.path.join(save_path, "evaluations.jsonl")):
    if True:
        solutions = list(stream_jsonl(os.path.join(save_path, "solutions.jsonl")))
        evaluations = thread_map(
            task.process_results,
            solutions,
            max_workers = args.num_workers,
            desc = "Evaluating Solutions"
        )
        evaluations = sorted(evaluations, key = lambda x: (x['task_id'], x['completion_id']))
        write_jsonl(os.path.join(save_path, "evaluations.jsonl"), evaluations)

    # calculate pass@k
    # if not os.path.exists(os.path.join(save_path, "results.jsonl")):
    if True:
        evaluations = list(stream_jsonl(os.path.join(save_path, "evaluations.jsonl")))
        results = calculate_pass_at_k(evaluations, args.num_samples, args.list_k)
        write_jsonl(os.path.join(save_path, "results.jsonl"), results)


if __name__ == "__main__":
    main()