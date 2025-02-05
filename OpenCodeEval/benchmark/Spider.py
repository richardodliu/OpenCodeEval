import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT))])

from benchmark.base import Benchmark
from utils import refine_text, program_extract, markdown_extract
from eval.sql_eval import execute_model

class Spider(Benchmark):

    name: str = "Spider"
    path: str = None
    database: str = None
    
    dev_path = os.path.abspath(os.path.join(ROOT, "../data/spider-dev/spider-dev.json"))
    dev_database = os.path.abspath(os.path.join(ROOT, "../data/spider-dev/database"))

    def __init__(self,
                 name: str = "SpiderDev",
                 timeout: float = 3.0,
                 prompt_type: str = "Instruction"): 
        super().__init__()
        
        self.name = name
        self.timeout = timeout
        self.prompt_type = prompt_type

        if self.name == "SpiderDev":
            self.path = self.dev_path
            self.database = self.dev_database

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

        result, passed = execute_model(solution['solution'],
                                       task_data['output'],
                                       db_path,
                                       self.timeout,
                                       self.name)
        
        return dict(
            task_id = solution['task_id'],
            completion_id = solution['completion_id'],
            passed = passed,
            result = result,
            solution = solution['solution']
        )