import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

from src.benchmark.base import Benchmark, PYTHON_IMPORTS
from src.sanitize import sanitize
from src.utils import refine_text, stream_jsonl
from src.eval.execution import check_correctness

class LeetCode(Benchmark):

    name: str = "LeetCode" 
    path = os.path.abspath(os.path.join(ROOT, "../data/20240121-Jul.jsonl"))

    def __init__(self,
                 name: str = "LeetCode",
                 timeout = 3.0,
                 prompt_type = "Completion"): 
        super().__init__()
        
        self.name = name
        self.timeout = timeout
        self.prompt_type = prompt_type

        self.tasks = self.get_task()
        self.imports = PYTHON_IMPORTS

        self.leetcode_prompt = "\nfrom typing import *\n\nfrom functools import *\nfrom collections import *\nfrom itertools import *\nfrom heapq import *\nfrom bisect import *\nfrom string import *\nfrom operator import *\nfrom math import *\nimport math\nimport datetime\ninf = float('inf')\n"

    def get_task(self):
        """
        Get the task data from the jsonl file into a dictionary.
        """

        tasks = {}
        
        for task_data in stream_jsonl(filename=self.path):

            task_id = int(task_data["meta"]["questionId"])
            tasks[task_id] = task_data
        
        return tasks
        
    def get_prompt(self):
        """
        Builds the prompt for the LM to generate from.
        """

        prompts = []
        for task_id, task_data in self.tasks.items():

            if self.prompt_type == "Completion":
                prompt = task_data['prompt']
            elif self.prompt_type == "Instruction":
                prompt = task_data['prompt_sft']

            prompts.append(
                dict(
                    task_id = task_id,
                    prompt = refine_text(prompt)
                )
            )

        return prompts

    def postprocess_generation(self, generation):
        """
        Postprocess the generations.
        """

        return dict(
            task_id = generation['task_id'],
            completion_id = generation['completion_id'],
            solution = sanitize(generation['completion'])
        )
    
    def process_results(self, solution):
        """
        Takes the list of LM generations and evaluates them against the test cases
        """

        task_data = self.tasks[solution['task_id']]

        code =  (
                    "\n".join(self.imports) + "\n\n"
                    + self.leetcode_prompt + "\n\n"
                    + solution['solution'] + "\n\n"
                    + task_data['test']
                )
        
        result = check_correctness(solution['task_id'],
                                   solution['completion_id'],
                                   code,
                                   self.timeout)
        
        return result