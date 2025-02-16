import os
from typing import Literal

from OpenCodeEval.benchmark.base import Benchmark, PYTHON_STOP, PYTHON_IMPORTS
from OpenCodeEval.utils import refine_text, stream_jsonl, program_extract
from OpenCodeEval.eval.func_eval import check_correctness
from OpenCodeEval.eval.sanitize import sanitize

class LiveCodeBench(Benchmark):

    name: str = "LiveCodeBench"
    path: str = None

    platform_dict = dict(
        atcoder = 1,
        codeforces = 2,
        leetcode = 3,
    )

    def __init__(
        self,
        split: Literal["v1", "v2", "v3", "v4", "v5"] = "v5",
        time_out: float = 3.0,
        prompt_type: str = "Instruction"
    ):

        super().__init__()

        self.path = os.path.join(self.path, self.name)

        self.tasks = self.get_task()

    def get_task_id(self, data):
        """
        Get the task id for the task.
        """

        from datetime import datetime

        date_id = datetime.fromisoformat(data['contest_date'])

        # refromat the date to YYYYMMDD
        date_id = date_id.strftime("%Y%m%d")

        if data['platform'] == 'atcoder':

            paltform_id = "1"
            contest, letter = data['question_id'].split('_')
            contest = ''.join(token for token in contest if token.isdigit())
            contest = contest.zfill(4)
                
            task_id =  paltform_id + contest + str(ord(letter) - ord('a') + 1)

        elif data['platform'] == 'codeforces':
            paltform_id = "2"
            contest, letter = data['question_id'].split('_')
            task_id =  paltform_id + contest + str(ord(letter) - ord('A') + 1)

        elif data['platform'] == 'leetcode':
            paltform_id = "3"
            task_id = paltform_id + data['question_id'] + "0"
        
        else:
            logger.error(f"Invalid platform: {data['platform']}")

        return int(task_id)

    def get_task(self):
        """
        Get the task data from the jsonl file into a dictionary.
        """
        
        version = int(self.split.split('v')[1])

        for i in range(1, version + 1):
            