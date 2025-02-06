import os
import json

from typing import Literal

from OpenCodeEval.benchmark.base import Benchmark
from OpenCodeEval.utils import refine_text, program_extract, markdown_extract
from OpenCodeEval.eval.sql_test import execute_sql

class Spider(Benchmark):

    name: str = "Spider"

    def __init__(
        self,
        split: Literal["train", "dev"] = "dev",
        timeout: float = 3.0,
        prompt_type: str = "Instruction"
    ):
    
        super().__init__()
        
        self.split = split
        self.timeout = timeout
        self.prompt_type = prompt_type

        self.path = os.path.join(self.path, f"{self.name}/{self.split}/data.jsonl")
        self.database = os.path.join(self.database, f"{self.name}/{self.split}/database")

        self.tasks = self.get_task()

    def get_task(self):
        """
        Get the task data from the json file into a dictionary.
        """

        with open(self.path) as f:
            task_set =  json.load(f)

        tasks = {}
        for task_data in task_set:
            tasks[int(task_data['id'])] = task_data
            
        return tasks

    def get_prompt(self):
        """
        Builds the prompt for the LM to generate from.
        """

        prompts = []
        
        
        for task_id, task_data in self.tasks.items():
            
            prompt = task_data['instruction']
            
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

        solution = ' '.join(program_extract(
            text = generation['completion'],
            program = 'sql', 
            mode = 'last').splitlines()
        )

        if solution == "":
            solution = ' '.join(markdown_extract(
                text = generation['completion'],
                mode = 'last').splitlines()
            )

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

        db_path = self.database + f"/{task_data['db_id']}/{task_data['db_id']}.sqlite"

        result, passed = execute_sql(
            solution['solution'],
            task_data['output'],
            db_path,
            self.timeout,
            "exact_match"
        )
        
        return dict(
            task_id = solution['task_id'],
            completion_id = solution['completion_id'],
            passed = passed,
            result = result,
            solution = solution['solution']
        )