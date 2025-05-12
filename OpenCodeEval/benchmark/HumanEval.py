import os
from typing import Literal

from OpenCodeEval.benchmark.base import Benchmark, PYTHON_STOP, PYTHON_IMPORTS
from OpenCodeEval.utils import refine_text, stream_jsonl, program_extract
from OpenCodeEval.eval.func_eval import check_correctness
from OpenCodeEval.eval.sanitize import sanitize

class HumanEval(Benchmark):

    name: str = "HumanEval"

    imports_code = PYTHON_IMPORTS
    chat_stop = PYTHON_STOP
    base_stop = ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]

    def __init__(
        self,
        split: Literal["base", "hard"] = "base",
        time_out: float = 3.0,
        prompt_type: str = "Completion"
    ):

        super().__init__()

        self.split = split
        self.time_out = time_out
        self.prompt_type = prompt_type

        self.path = os.path.join(self.path, f"{self.name}/{self.split}.jsonl")

        self.tasks = self.get_task()

    def get_task(self):
        """
        Get the task data from the jsonl file into a dictionary.
        """

        tasks = {}
        
        for task_data in stream_jsonl(filename=self.path):

            task_id = int(task_data["task_id"].split("/")[-1])
            
            tasks[task_id] = task_data
        
        return tasks

    def get_prompt(self):
        """
        Builds the prompt for the LM to generate from.
        """

        assert self.prompt_type == "Completion", "Prompt type must be Completion for HumanEval"

        prompts = []
        for task_id, task_data in self.tasks.items():
            prompts.append(
                dict(
                    task_id = task_id,
                    prompt = refine_text(task_data['prompt'])
                )
            )
        return prompts

    def postprocess_generation(self, generation):
        """
        Postprocess the generations.
        """

        entry_point = self.tasks[generation['task_id']]["entry_point"]

        try:
            completion = '\n'.join(generation['completion'].splitlines()[-200:])

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

        code = (
            "\n".join(self.imports_code) + "\n"
            + task_data["prompt"] + "\n"
            + "    pass\n" + "\n"
            + solution['solution'] + "\n"
            + task_data['test'] + "\n"
            + f"check({task_data['entry_point']})"
            )

        result = check_correctness(
            solution['task_id'],
            solution['completion_id'],
            code,
            self.time_out
        )
        
        return result