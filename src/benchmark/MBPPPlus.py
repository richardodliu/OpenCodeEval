import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT))])

from benchmark.base import Benchmark, PYTHON_STOP, PYTHON_IMPORTS
from sanitize import sanitize
from eval.execution import check_correctness
from utils import refine_text, stream_jsonl

class MBPPPlus(Benchmark):

    name: str = "MBPPPlus"
    path: str = os.path.abspath(os.path.join(ROOT, "../data/MBPPPlus.jsonl"))

    imports_code = PYTHON_IMPORTS
    chat_stop = PYTHON_STOP
    base_stop = ['\n"""', "\nassert"]

    general_stop_words = [  "<|endoftext|>",
                            "<|endofmask|>",
                            "</s>",
                            "\nif __name__",
                            "\ndef main(",
                            "\nprint(",
                            '\n```\n']
    
    completion_stop_words = [   "\ndef ",
                                "\nclass ",
                                "\nimport ",
                                "\nfrom ",
                                "\nassert " ]

    def __init__(self,
                 name: str = "MBPPPlus",
                 num_samples: int = 1,
                 num_workers: int = 16,
                 timeout: float = 3.0,
                 prompt_type: str = "Instruction"):
        
        super().__init__()
        self.name = name
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.timeout = timeout
        self.prompt_type = prompt_type

    def get_task(self):

        return list(stream_jsonl(filename = self.path))
    
    def format_prompt(self, 
                     promblem: str,
                     test: str,
                     ) -> str:
        # promblem = f"You are an expert Python programmer, and here is your task:\n{promblem}\n"
        # test = f"Your code should pass the test:\n{test}\n"
        # prompt = promblem + test
        prompt = promblem + "\n" + test
        return prompt
    
    def get_prompt(self):

        assert self.prompt_type == "Instruction", f"Prompt type must be Instruction for {self.name}"

        task_set = self.get_task()
        prompts = []
        for task_data in task_set:
            prompt = self.format_prompt(task_data["text"], task_data["test_list"][0])
            prompts.append(refine_text(prompt))

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
        Process the solutions.
        """
        return dict(
            task_id = solution['task_id'],
            solution_id = solution['solution_id'],
            solution = solution['solution']
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