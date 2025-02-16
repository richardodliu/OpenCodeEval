import os

from loguru import logger

from tqdm import tqdm
from typing import List, Dict

from OpenCodeEval.backend.base import Generator, make_chat_template

from OpenCodeEval.utils import refine_text

class SglangGenerator(Generator):

    def __init__(
        self,
        model_name: str,
        tokenizer_name: str = None,
        model_type: str = "Instruction",
        dtype: str = "bfloat16",
        batch_size : int = 1,
        temperature : float = 0.0,
        top_p : float = 1.0,
        max_tokens : int = 1024,
        num_samples: int = 200,
        num_gpus: int = 1,
        trust_remote_code: bool = True,
    ) -> None:

        super().__init__(model_name)

        print("Initializing a decoder model in sglang: {} ...".format(model_name))
        self.tokenizer_name = tokenizer_name if tokenizer_name is not None else model_name
        self.model_type = model_type
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.num_samples = num_samples
        self.dtype = dtype
        self.num_gpus = num_gpus
        self.trust_remote_code = trust_remote_code

        from sglang import Engine

        self.model = Engine(
            model_path = self.model_name,
            tokenizer_path = self.tokenizer_name,
            dtype = self.dtype,
            tp_size = self.num_gpus,
            trust_remote_code = self.trust_remote_code
        )

        self.tokenizer = self.model.tokenizer_manager.tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.sampling_params = dict(
            n = self.num_samples,
            temperature = self.temperature,
            max_new_tokens = self.max_tokens,
            top_p = self.top_p,
        )
    
    def is_chat(self) -> bool:
        if self.model_type == "Chat":
            if self.tokenizer.chat_template is None:
                logger.error("When the model type is Chat, the chat template must be set for the tokenizer.")
            else:
                return True
        else:
            return False
        
    def set_stop(self, eos: List[str]):
        self.sampling_params['stop'] = eos
        
    def release_memory(self):

        import gc
        import torch

        from sglang.srt.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

        self.model.shutdown()

        # destroy_model_parallel
        # destroy_distributed_environment()

        # del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def generate(
            self,
            prompt_set: List[Dict],
            response_prefix: str = "",
            response_suffix: str = ""
        ) -> List[str]:

        if self.is_chat():
            for prompt in prompt_set:
                prompt['prompt'] = make_chat_template(
                    prompt = prompt['prompt'],
                    response_prefix = response_prefix,
                    is_chat = self.is_chat(),
                    tokenizer = self.tokenizer
                )

        logger.info("Example Prompt:\n{}", prompt_set[0]['prompt'])

        generation_set = []

        for batch_start in tqdm(range(0, len(prompt_set), self.batch_size)):

            batch_prompt = prompt_set[batch_start : batch_start + self.batch_size]
            batch_outputs = self.model.generate(
                [prompt['prompt'] for prompt in batch_prompt],
                self.sampling_params
            )

            batch_outputs = [
                [item["text"] for item in batch_outputs[i:i + self.num_samples]]
                for i in range(0, len(batch_outputs), self.num_samples)
            ]

            for prompt, output in zip(batch_prompt, batch_outputs):

                # Process completions with prefix/suffix and refine text based on chat mode
                completions = [
                    refine_text(
                        f"{prompt['prompt']}\n\n{response_prefix}{completion}{response_suffix}"
                        if not self.is_chat()
                        else f"{response_prefix}{completion}{response_suffix}"
                    )
                    for completion in output
                ]

                for completion_id, completion in enumerate(completions):
                    generation_set.append({
                        'task_id': prompt['task_id'],
                        'completion_id': completion_id,
                        'completion': completion
                    })

        if len(generation_set) != len(prompt_set) * self.num_samples:
            logger.error("Number of generations does not match the expected number.")

        self.release_memory()

        return generation_set