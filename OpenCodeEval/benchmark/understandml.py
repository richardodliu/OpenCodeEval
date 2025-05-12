import os

from OpenCodeEval.benchmark.base import Benchmark, PYTHON_STOP, PYTHON_IMPORTS
from OpenCodeEval.utils import program_extract, stream_jsonl
from OpenCodeEval.eval.func_eval import check_correctness
from OpenCodeEval.eval.sanitize import sanitize

from typing import List, Literal

from typing import *
from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *

import numpy as np
from numpy import *

import datetime
import copy

inf = float('inf')

if os.environ.get("MAX_LINES"):
    MAX_LINES = int(os.environ.get("MAX_LINES"))
else:
    MAX_LINES = 200

def base_prompt(data):
    prompt = 'You are an expert Python programmer, and here is your task:\n'
    prompt = prompt + f'# Task: {data["title"]}\n'
    prompt = prompt + f'# Description:\n{data["description"]}\n'
    # prompt = prompt + f'# Examples:\n'
    # for example_idx, (example, reasoning) in enumerate(zip(data["examples"], data["reasoning"])):
    #     prompt = prompt + f'## Example {example_idx + 1}:\n'
    #     prompt = prompt + f'### Input:\n{example["input"]}\n'
    #     prompt = prompt + f'### Output:\n{example["output"]}\n'
    #     prompt = prompt + f'### Reasoning:\n{reasoning}\n'
    input_code = (data["import_code"] + "\n" + data["starter_code"]).strip()
    prompt = prompt + f'# Your code should start with:\n```python\n{input_code}\n```\n'
    if data['output_constrains'].strip():
        prompt = prompt + f'# Output Constraints:\n{data["output_constrains"].strip()}\n'

    return prompt



class understandml(Benchmark):

    name: str = "understandml"

    imports_code = PYTHON_IMPORTS
    chat_stop = PYTHON_STOP
    base_stop = ['\n"""', "\nassert"]

    def __init__(
        self,
        split: Literal["human", "model"] = "model",
        time_out: float = 3.0,
        prompt_type: str = "Instruction"
    ):

        super().__init__()

        self.split = split
        self.time_out = time_out
        self.prompt_type = prompt_type

        self.path = os.path.join(self.path, f"{self.name}/{self.split}_benchmark.jsonl")
        self.tasks = self.get_task()
    
    def get_task(self):
        """
        Get the task data from the jsonl file into a dictionary.
        """

        tasks = {}
        
        for task_data in stream_jsonl(filename = self.path):
            task_id = int(task_data["id"])
            
            tasks[task_id] = task_data
        
        return tasks
    
    def get_prompt(self):
        """
        Builds the prompt for the LM to generate from.
        """

        assert self.prompt_type == "Instruction", "Prompt type must be Instruction for mbpp"

        prompts = []

        for task_id, task_data in self.tasks.items():
                
            prompt = base_prompt(task_data)
            prompts.append({
                'task_id': task_id,
                'prompt': prompt
            })

        return prompts

    def postprocess_generation(self, generation):
        """
        Postprocess the generations.
        """

        entry_point = self.tasks[generation['task_id']]["entry_point"]

        try:
            completion = '\n'.join(generation['completion'].splitlines()[-MAX_LINES:])

            if '</think>' in completion:
                completion = completion.split('</think>')[1]
            
            solution = sanitize(completion, entry_point)
        except Exception:
            solution = program_extract(generation['completion'], program="python", mode="all")

        result = dict(
            task_id = generation['task_id'],
            completion_id = generation['completion_id'],
            solution = solution
        )

        return result
    
    def process_results(self, solution):
        """
        Takes the list of LM generations and evaluates them against the test cases
        """

        task_data = self.tasks[solution['task_id']]

        code =  (
                    task_data['import_code'] + "\n"
                    + solution['solution'] + "\n"
                    + "\n".join(task_data['test_cases'])
                )
        
        result = check_correctness(solution['task_id'],
                                   solution['completion_id'],
                                   code,
                                   self.time_out)
        
        return result