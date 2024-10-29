import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT))])

from tqdm import tqdm
from typing import Literal
from concurrent.futures import ProcessPoolExecutor, as_completed

from benchmark.base import Benchmark
from sanitize import sanitize
from utils import refine_text, stream_jsonl
from eval.unittest_execution import check_correctness

class BigCodeBench(Benchmark):

    name: str = "BigCodeBench"

    fullset_path = os.path.abspath(os.path.join(ROOT, "../data/BigCodeBench.jsonl"))
    subset_path = os.path.abspath(os.path.join(ROOT, "../data/BigCodeBench_Hard.jsonl"))

    def __init__(self,
                 name: str = "BigCodeBench",
                 timeout:float = 10.0,
                 prompt_type: Literal["Completion", "Instruction"] = "Completion"
                 ):
        
        super().__init__()
        
        self.name = name
        self.timeout = timeout
        self.prompt_type = prompt_type

        if self.name == "BigCodeHard":
            self.path = self.subset_path
        elif self.name == "BigCodeBench":
            self.path = self.fullset_path

    def get_task(self):
        
        return list(stream_jsonl(filename = self.path))

    def get_prompt(self):
        """
        Get the prompt for the test set.
        """
        task_set = self.get_task()

        if self.prompt_type == "Completion":
            return [refine_text(data["complete_prompt"]) for data in task_set]
        elif self.prompt_type == "Instruction":
            return [refine_text(data["instruct_prompt"]) for data in task_set]

    def postprocess_generation(self, generation_set):
        """
        Postprocess the generations.
        """
        task_set = self.get_task()
        assert len(generation_set) == len(task_set), f"Num generations : {len(generation_set)} not match Test set length: {len(task_set)}"

        solution_set = []
        for index, sample_generations in tqdm(enumerate(generation_set), total=len(generation_set), desc="Postprocessing"):

            assert len(sample_generations) == self.num_samples, \
                f"Num generations : {len(sample_generations)} not match Num samples: {self.num_samples}"
            solution_set.append([sanitize(single_generation, task_set[index]["entry_point"]) for single_generation in sample_generations])

        return solution_set

    def process_results(self, solutions):

        eval_args = []
        task_set = self.get_task()

        for task_data, solutions_list in zip(task_set, solutions):
            assert len(solutions_list) == self.num_samples, f"Num completions : {len(solutions_list)} not match Num samples: {self.num_samples}"
            task_id = int(task_data["task_id"].split('/')[-1])
            for solution_id, solution_data in enumerate(solutions_list):
                solution = (
                    task_data["code_prompt"] + "\n" 
                    + "    pass\n" + "\n"
                    + solution_data + "\n"
                )

                eval_args.append({
                    "task_id": task_id,
                    "solution_id": solution_id,
                    "solution": solution,
                    "test": task_data["test"]
                })

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for eval in eval_args:
                args = (eval['task_id'], eval['solution_id'], eval['solution'], eval['test'], self.timeout)
                future = executor.submit(check_correctness, *args)
                futures.append(future)
            
            evalution_set = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Completing tasks"):
                result = future.result()
                evalution_set.append(result)

        return evalution_set