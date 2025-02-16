import os
from typing import Literal
from loguru import logger

from OpenCodeEval.benchmark.base import Benchmark, PYTHON_IMPORTS, LEETCODE_IMPORTS, PYTHON_STOP
from OpenCodeEval.utils import refine_text, stream_jsonl
from OpenCodeEval.eval.func_eval import check_correctness
from OpenCodeEval.eval.sanitize import sanitize
class LeetCode(Benchmark):

    name: str = "LeetCode"

    imports_code = PYTHON_IMPORTS + LEETCODE_IMPORTS
    chat_stop = PYTHON_STOP
    base_stop = ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]

    def __init__(
        self,
        split: Literal["contest", "train", "validation", "test"] = "contest",
        time_out: float = 3.0,
        prompt_type: Literal["Completion", "Instruction"] = "Instruction"
    ):

        super().__init__()
        
        self.name = name
        self.split = split
        self.time_out = time_out

        self.prompt_type = prompt_type
        if self.split != "contest" and self.prompt_type == "Completion":
            logger.error(f"Completion prompt type not support {self.split} split")

        self.path = os.path.join(self.path, f"{self.name}/{self.split}.jsonl")
        self.tasks = self.get_task()

    def get_task(self):
        """
        Get the task data from the jsonl file into a dictionary.
        """

        tasks = {}
        
        for task_data in stream_jsonl(filename=self.path):

            if self.split == "contest":
                task_id = int(task_data["meta"]["questionId"])
            else:
                task_id = int(task_data["meta"]["question_id"])
            tasks[task_id] = task_data
        
        return tasks
        
    def get_prompt(self):
        """
        Builds the prompt for the LM to generate from.
        """

        prompts = []
        for task_id, task_data in self.tasks.items():

            if self.split == "contest":
                if self.prompt_type == "Completion":
                    prompt = task_data['prompt']
                elif self.prompt_type == "Instruction":
                    prompt = task_data['prompt_sft']
            else:
                prompt = task_data['meta']['query']

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
            solution = sanitize(
                text = generation['completion'],
                entrypoint = "Solution",
            )
        )
    
    def process_results(self, solution):
        """
        Takes the list of LM generations and evaluates them against the test cases
        """

        task_data = self.tasks[solution['task_id']]

        if self.split == "contest":
            code = (
                "\n".join(self.imports_code) + "\n\n"
                + solution['solution'] + "\n\n"
                + task_data['test']
                )
        else:
            code = (
                "\n".join(self.imports_code) + "\n\n"
                + task_data['meta']['lang_code'] + "\n"
                + "        pass\n" + "\n"
                + solution['solution'] + "\n"
                + task_data['test'] + "\n"
                + f"check({task_data['entry_point']})"
                )
        
        result = check_correctness(solution['task_id'],
                                   solution['completion_id'],
                                   code,
                                   self.time_out)
        
        return result