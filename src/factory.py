import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT))])

from benchmark.HumanEval import HumanEval
from benchmark.MBPP import MBPP
from benchmark.MBPPPlus import MBPPPlus
from benchmark.LeetCode import LeetCode
from benchmark.BigCodeBench import BigCodeBench
from benchmark.Bird import Bird
from benchmark.Spider import Spider

from backend.vllm import VllmGenerator
from backend.openai import OpenaiGenerator

class BenchmarkFactory:
    @staticmethod
    def get_task(args):
        if args.task == "HumanEval" or args.task == "HumanEvalPlus":
            return HumanEval(name = args.task,
                             timeout = args.time_out,
                             prompt_type = args.prompt_type)
        elif args.task == "MBPP":
            return MBPP(name = args.task,
                        prompt_type = args.prompt_type)
        elif args.task == "MBPPPlus" or args.task == "MBPPBase":
            return MBPPPlus(name = args.task,
                            prompt_type = args.prompt_type)
        elif args.task == "LeetCode":
            return LeetCode(name = args.task,
                            prompt_type = args.prompt_type)
        elif args.task == "BigCodeBench" or args.task == "BigCodeHard":
            return BigCodeBench(name = args.task,
                                prompt_type = args.prompt_type)
        elif args.task == "BirdDev":
            return Bird(name = args.task,
                        timeout = args.time_out,
                        prompt_type = args.prompt_type)
        elif args.task == "SpiderDev":
            return Spider(name = args.task,
                          timeout = args.time_out,
                          prompt_type = args.prompt_type
            )
        else: 
            raise ValueError("Unknown Task type")

class BackendFactory:
    @staticmethod
    def get_backend(args):
        if args.backend == "vllm":
            return VllmGenerator(
                model_name = args.model_name,
                model_type = args.model_type,
                tokenizer_name = args.tokenizer_name,
                num_gpus = args.num_gpus,
                batch_size = args.batch_size,
                temperature = args.temperature,
                num_samples = args.num_samples,
                trust_remote_code = args.trust_remote_code,
                max_tokens = args.max_tokens)
        elif args.backend == "openai":
            return OpenaiGenerator(model_name = args.model_name,
                                   model_type = args.model_type,
                                   temperature = args.temperature,
                                   max_tokens = args.max_tokens,
                                   num_samples = args.num_samples,
                                   batch_size = args.batch_size)
        else:
            raise ValueError("Unknown Backend type")
