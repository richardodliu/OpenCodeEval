import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT))])

import gc
import torch
from loguru import logger
from tqdm import tqdm
from typing import List, Dict
from backend.base import Generator
from utils import refine_text

class SglangGenerator(Generator):
    def __init__(self,
                 model_name: str, 
                 tokenizer_name: str = None,
                 model_type: str = "Instruction",
                 dtype: str = "bfloat16",
                 batch_size: int = 1,
                 temperature: float = 0.0,
                 max_tokens: int = 1024,
                 num_samples: int = 200,
                 num_gpus: int = 1,
                 trust_remote_code: bool = True
                ) -> None:
        
        import sglang as sgl

        super().__init__(model_name)
        
        print(f"Initializing a decoder model: {model_name} ...")
        self.tokenizer_name = tokenizer_name if tokenizer_name else model_name
        self.model_type = model_type
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_samples = num_samples
        self.dtype = dtype
        self.num_gpus = num_gpus
        self.trust_remote_code = trust_remote_code

        # 初始化sglang模型
        self.model = sgl.Engine(
            model_name=self.model_name,
            tensor_parallel_size=self.num_gpus
        )
        self.tokenizer = self.model.get_tokenizer()

    def make_chat_template(self, prompt: str, response_prefix: str = "") -> str:
        if self.is_chat():
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            ) + response_prefix
            if self.tokenizer.bos_token and prompt.startswith(self.tokenizer.bos_token):
                prompt = prompt[len(self.tokenizer.bos_token):]
            return prompt
        else:
            return '''You are a helpful programming assistant.
### Instruction:
{}
### Response:
'''.format(prompt.strip()).lstrip() + response_prefix

    def is_chat(self) -> bool:
        if self.model_type == "Chat":
            assert self.model.get_tokenizer().chat_template is not None
            return True
        return False

    def release_memory(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def generate(self,
                prompt_set: List[Dict],
                eos: List[str] = None,
                response_prefix: str = "",
                response_suffix: str = ""
               ) -> List[str]:
        
        if self.is_chat():
            for prompt in prompt_set:
                prompt['prompt'] = self.make_chat_template(prompt['prompt'])

        logger.info("示例提示:\n{}", prompt_set[0]['prompt'])

        generation_set = []

        for batch_start in tqdm(range(0, len(prompt_set), self.batch_size)):
            batch_prompt = prompt_set[batch_start : batch_start + self.batch_size]
            
            # 使用sglang的生成API
            batch_outputs = []
            for prompt in batch_prompt:
                state = self.model.create_state()
                state.llm(prompt['prompt'], 
                         temperature=self.temperature,
                         max_tokens=self.max_tokens,
                         num_return_sequences=self.num_samples,
                         stop=eos)
                batch_outputs.append(state.get_outputs())

            for prompt, outputs in zip(batch_prompt, batch_outputs):
                completions = [
                    refine_text(
                        f"{prompt['prompt']}\n\n{response_prefix}{output}{response_suffix}"
                        if not self.is_chat()
                        else f"{response_prefix}{output}{response_suffix}"
                    )
                    for output in outputs
                ]

                for completion_id, completion in enumerate(completions):
                    generation_set.append({
                        'task_id': prompt['task_id'],
                        'completion_id': completion_id,
                        'completion': completion
                    })

        assert len(generation_set) == len(prompt_set) * self.num_samples, "生成数量与预期不符"

        self.release_memory()

        return generation_set