import os

from OpenCodeEval.benchmark.base import Benchmark, PYTHON_STOP, PYTHON_IMPORTS
from OpenCodeEval.utils import refine_text, stream_jsonl
from OpenCodeEval.eval.func_eval import check_correctness
from OpenCodeEval.eval.sanitize import sanitize

from typing import List, Literal

class mbpp(Benchmark):

    name: str = "mbpp"

    imports_code = PYTHON_IMPORTS
    chat_stop = PYTHON_STOP
    base_stop = ['\n"""', "\nassert"]

    def __init__(
        self,
        split: Literal["full", "sanitized"] = "full",
        time_out: float = 3.0,
        prompt_type: str = "Instruction"
    ):

        super().__init__()

        self.split = split
        self.time_out = time_out
        self.prompt_type = prompt_type

        self.path = os.path.join(self.path, f"{self.name}/{self.split}.jsonl")
        self.tasks = self.get_task()

        self.few_shots_prompt = self.get_few_shots_prompts() if split == "full" else ""
    
    def get_task(self):
        """
        Get the task data from the jsonl file into a dictionary.
        """

        tasks = {}
        
        for task_data in stream_jsonl(filename = self.path):
            task_id = int(task_data["task_id"])

            task_data['text'] = refine_text(task_data['text'])
            
            tasks[task_id] = task_data
        
        return tasks

    def fewshot_examples(self):

        few_shots_start = 1
        few_shots_end = 4

        few_shots = []

        for task_id, task_data in self.tasks.items():
            if task_id >= few_shots_start and task_id < few_shots_end:
                few_shots.append(task_data)
        
        return few_shots
    
    def format_prompt(self,
                      promblem: str,
                      tests: List[str],
                      code: str = None
                    ) -> str:
        promblem = f"You are an expert Python programmer, and here is your task:\n{promblem}"
        test = "\n".join(tests)
        test = f"Your code should pass these tests:\n{test}\n"
        prompt = promblem + test
        if code:
            code = refine_text(code)
            code = f"\n```python\n{code}\n```\n"
            prompt = prompt + code
        else:
            prompt = prompt + "\n```python\n"
        return prompt
    
    def get_few_shots_prompts(self):
        
        few_shots_prompts = []
        for few_shot in self.fewshot_examples():
            few_shots_prompts.append(self.format_prompt(few_shot["text"], few_shot["test_list"], few_shot["code"]))

        return '\n'.join(few_shots_prompts)
    
    def get_prompt(self):
        """
        Builds the prompt for the LM to generate from.
        """

        assert self.prompt_type == "Instruction", "Prompt type must be Instruction for mbpp"

        if self.split == "full":
            test_start = 10
            test_end = 510
        elif self.split == "sanitized":
            test_start = 0
            test_end = 974

        prompts = []

        for task_id, task_data in self.tasks.items():
            if task_id >= test_start and task_id < test_end:
                
                prompt = self.few_shots_prompt + '\n' + self.format_prompt(task_data["text"], task_data["test_list"])
                prompts.append({
                    'task_id': task_id,
                    'prompt': prompt
                })

        return prompts

    def postprocess_generation(self, generation):
        """
        Postprocess the generations.
        """

        if generation['completion'].startswith(self.few_shots_prompt):
            generation['completion'] = generation['completion'][len(self.few_shots_prompt):]

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
                    "\n".join(self.imports_code) + "\n"
                    + task_data['test_setup_code'] + "\n"
                    + solution['solution'] + "\n"
                    + "\n".join(task_data['test_list'])
                )
        
        result = check_correctness(solution['task_id'],
                                   solution['completion_id'],
                                   code,
                                   self.time_out)
        
        return result
