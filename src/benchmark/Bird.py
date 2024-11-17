import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT))])

from tqdm import tqdm
from typing import Literal

from benchmark.base import Benchmark
from utils import refine_text
from sql_utils import generate_schema_prompt

class Bird(Benchmark):

    name: str = "Bird"
    path: str = None
    database: str = None
    
    dev_path = os.path.abspath(os.path.join(ROOT, "../data/bird/dev/dev.json"))
    dev_database = os.path.abspath(os.path.join(ROOT, "../data/bird/dev/dev_databases"))

    def __init__(self,
                 name: str = "BirdDev",
                 timeout: float = 3.0,
                 prompt_type: str = "Instruction"): 
        super().__init__()
        
        self.name = name
        self.timeout = timeout
        self.prompt_type = prompt_type

        if self.name == "BirdDev":
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
            tasks[int(task_data['question_id'])] = task_data
            
        return tasks

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""

        ini_table = "CREATE TABLE singer\n(\n    singer_id         TEXT not null\n        primary key,\n    nation       TEXT  not null,\n    sname       TEXT null,\n    dname       TEXT null,\n    cname       TEXT null,\n    age    INTEGER         not null,\n    year  INTEGER          not null,\n    birth_year  INTEGER          null,\n    salary  REAL          null,\n    city TEXT          null,\n    phone_number   INTEGER          null,\n--     tax   REAL      null,\n)"
        ini_prompt = "-- External Knowledge: age = year - birth_year;\n-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above.\n-- How many singers in USA who is older than 27?\nThe final SQL is: Let's think step by step."
        ini_cot_result = "1. referring to external knowledge, we need to filter singers 'by year' - 'birth_year' > 27; 2. we should find out the singers of step 1 in which nation = 'US', 3. use COUNT() to count how many singers. Finally the SQL is: ```sql\nSELECT COUNT(*) FROM singer WHERE year - birth_year > 27;\n```"
        
        one_shot_demo = ini_table + '\n' + ini_prompt + '\n' + ini_cot_result
        
        return one_shot_demo

    def get_prompt(self):
        """
        Builds the prompt for the LM to generate from.
        """

        prompts = []
        for task_id, task_data in self.tasks.items():
            db_id = task_data['db_id']
            question = task_data['question']
            knowledge = task_data['evidence']
            
            db_path = self.database + f"/{db_id}/{db_id}.sqlite"
            database_schema = generate_schema_prompt(db_path)

            knowledge_prompt = "-- External Knowledge: {}".format(knowledge)
            
            pattern_prompt_no_kg = "-- Using valid SQLite, answer the following questions for the tables provided above."
            pattern_prompt_kg = "-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above."

            question_prompt = "-- {}".format(question)
            
            result_prompt = knowledge_prompt + '\n' + pattern_prompt_kg + '\n' + question_prompt

            cot_prompt = "Generate the SQL after thinking step by step."
            
            prompt = database_schema + "\n\n" + result_prompt + "\n\n" + cot_prompt
            
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

        entry_point = self.tasks[generation['task_id']]["entry_point"]

        result = dict(
            task_id = generation['task_id'],
            completion_id = generation['completion_id'],
            solution = sanitize(generation['completion'], entry_point)
        )

        return result

    def process_results(self, solution):
        """
        Takes the list of LM generations and evaluates them against the test cases
        """

        task_data = self.tasks[solution['task_id']]

        code = ("\n".join(self.imports) + "\n"
                    + task_data["prompt"] + "\n"
                    + "    pass\n" + "\n"
                    + solution['solution'] + "\n"
                    + task_data['test'] + "\n"
                    + f"check({task_data['entry_point']})"
                )
        
        result = check_correctness(solution['task_id'],
                                   solution['completion_id'],
                                   code,
                                   self.timeout)
        
        return result