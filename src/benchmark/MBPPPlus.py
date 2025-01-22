import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

from src.benchmark.base import Benchmark, PYTHON_STOP, PYTHON_IMPORTS
from src.sanitize import sanitize
from src.utils import refine_text, stream_jsonl, program_extract
from src.eval.execution import check_correctness

class MBPPPlus(Benchmark):

    name: str = "MBPPPlus"
    path: str = os.path.abspath(os.path.join(ROOT, "../data/MBPPPlus.jsonl"))

    imports_code = PYTHON_IMPORTS
    chat_stop = PYTHON_STOP
    base_stop = ['\n"""', "\nassert"]
    # TODO: add more stop words, e.g. "\nif __name__", "\ndef main(", "\nprint(", '\n```\n']

    def __init__(self,
                 name: str = "MBPPPlus",
                 timeout: float = 3.0,
                 prompt_type: str = "Instruction"):
        
        super().__init__()
        
        self.name = name
        self.timeout = timeout
        self.prompt_type = prompt_type

        self.tasks = self.get_task()

    def get_task(self):
        """
        Get the task data from the jsonl file into a dictionary.
        """

        tasks = {}
        
        for task_data in stream_jsonl(filename=self.path):

            task_id = int(task_data["task_id"])
            
            tasks[task_id] = task_data
        
        return tasks
    
    def format_prompt(self, 
                     promblem: str,
                     test: str,
                     ) -> str:
        promblem = f"You are an expert Python programmer, and here is your task:\n{promblem}"
        test = f"Your code should pass the test:\n{test}"
        prompt = promblem + "\n" + test
        return prompt
    
    def get_prompt(self):
        """
        Builds the prompt for the LM to generate from.
        """

        assert self.prompt_type == "Instruction", "Prompt type must be Instruction for MBPP"

        prompts = []
        for task_id, task_data in self.tasks.items():
            prompts.append(
                dict(
                    task_id = task_id,
                    prompt = refine_text(self.format_prompt(task_data["text"], task_data["test_list"][0]))
                )
            )
        return prompts

    def postprocess_generation(self, generation):
        """
        Postprocess the generations.
        """

        entry_point = self.tasks[generation['task_id']]["entry_point"]

        try:
            solution = sanitize(generation['completion'])
            solution = solution.replace("func0", entry_point)
        except Exception:
            solution = program_extract(generation['completion'], program="python", mode="all")
        
        return dict(
            task_id = generation['task_id'],
            completion_id = generation['completion_id'],
            solution = solution
        )
    
    
    def process_results(self, solution):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """

        task_data = self.tasks[solution['task_id']]

        if self.name == "MBPPPlus":
            test_code = "\n".join(task_data['test_imports']) + "\n\n" + task_data['test']
        elif self.name == "MBPPBase":
            test_code = "\n".join(task_data['test_imports']) + "\n\n" + "\n".join(task_data['test_list'])

        code =  (
            "\n".join(self.imports_code) + "\n"
            + solution['solution'] + "\n"
            + test_code
        )

        result = check_correctness(solution['task_id'],
                                   solution['completion_id'],
                                   code,
                                   self.timeout)
        
        return result