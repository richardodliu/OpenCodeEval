import os
import json

from loguru import logger
from typing import Literal

from OpenCodeEval.benchmark.base import Benchmark
from OpenCodeEval.utils import refine_text, program_extract, markdown_extract, stream_jsonl
from OpenCodeEval.eval.sql_test import check_correctness

class Spider(Benchmark):

    name: str = "Spider"

    def __init__(
        self,
        split: Literal["train", "dev"] = "dev",
        time_out: float = 3.0,
        prompt_type: str = "Instruction"
    ):
    
        super().__init__()
        
        self.split = split
        self.time_out = time_out
        self.prompt_type = prompt_type

        if self.prompt_type == "Completion":
            logger.error("Completion prompt type not supported for Spider")

        self.database = os.path.join(self.path, f"{self.name}/{self.split}/database")
        self.path = os.path.join(self.path, f"{self.name}/{self.split}/data.jsonl")

        self.tasks = self.get_task()

    def get_task(self):
        """
        Get the task data from the json file into a dictionary.
        """

        tasks = {}
        
        for task_data in stream_jsonl(filename = self.path):

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

        db_path = os.path.join(self.database, f"{task_data['db_id']}/{task_data['db_id']}.sqlite")

        result, passed,sql_return = check_correctness(
            solution['solution'],
            task_data['output'],
            db_path,
            self.time_out,
            "exact_match"
        )
        
        return dict(
            task_id = solution['task_id'],
            completion_id = solution['completion_id'],
            passed = passed,
            result = result,
            solution = solution['solution'],
            sql_return = sql_return
        )