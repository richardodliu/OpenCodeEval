from loguru import logger

from OpenCodeEval.benchmark.HumanEval import HumanEval
from OpenCodeEval.benchmark.mbpp import mbpp
from OpenCodeEval.benchmark.MBPP import MBPP
from OpenCodeEval.benchmark.LeetCode import LeetCode
from OpenCodeEval.benchmark.BigCodeBench import BigCodeBench
from OpenCodeEval.benchmark.Bird import Bird
from OpenCodeEval.benchmark.Spider import Spider

from OpenCodeEval.benchmark.understandml import understandml

from OpenCodeEval.backend.vllm import VllmGenerator
from OpenCodeEval.backend.sglang import SglangGenerator
from OpenCodeEval.backend.openai import OpenaiGenerator
from OpenCodeEval.backend.hf import TransformerGenerator

class BenchmarkFactory:

    @staticmethod
    def get_task(args):
        task_map = {
            "HumanEval": HumanEval,
            "mbpp": mbpp,
            "MBPP": MBPP,
            "LeetCode": LeetCode,
            "BigCodeBench": BigCodeBench,
            "Bird": Bird,
            "Spider": Spider,
            "understandml": understandml
        }

        # Check if the task exists in the map
        if args.task not in task_map:
            logger.error(f"Unknown Task type: {args.task}")

        # Get the corresponding class
        task_class = task_map[args.task]

        return task_class(
            split = args.split,
            time_out = args.time_out,
            prompt_type = args.prompt_type
        )

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
                top_p = args.top_p,
                num_samples = args.num_samples,
                max_tokens = args.max_tokens,
                trust_remote_code = args.trust_remote_code,
            )

        elif args.backend == "sglang":
            return SglangGenerator(
                model_name = args.model_name,
                model_type = args.model_type,
                tokenizer_name = args.tokenizer_name,
                num_gpus = args.num_gpus,
                batch_size = args.batch_size,
                temperature = args.temperature,
                top_p = args.top_p,
                num_samples = args.num_samples,
                max_tokens = args.max_tokens,
                trust_remote_code = args.trust_remote_code,
            )

        elif args.backend == "openai":
            return OpenaiGenerator(
                model_name = args.model_name,
                model_type = args.model_type,
                batch_size = args.batch_size,
                temperature = args.temperature,
                top_p = args.top_p,
                num_samples = args.num_samples,
                max_tokens = args.max_tokens,
            )

        elif args.backend == "transformer":
            return TransformerGenerator(
                model_name = args.model_name,
                model_type = args.model_type,
                tokenizer_name = args.tokenizer_name,
                num_gpus = args.num_gpus,
                batch_size = args.batch_size,
                temperature = args.temperature,
                top_p = args.top_p,
                num_samples = args.num_samples,
                max_tokens = args.max_tokens,
                trust_remote_code = args.trust_remote_code,
            )
        else:
            logger.error(f"Unknown Backend type: {args.backend}")
