import os
import sys
import fla
import torch

from tqdm import tqdm
from loguru import logger
from typing import List, Dict

from OpenCodeEval.backend.base import Generator, make_chat_template

from OpenCodeEval.utils import refine_text

class TransformerGenerator(Generator):

    def __init__(
        self,
        model_name: str,
        tokenizer_name: str = None,
        model_type: str = "Instruction",
        dtype: str = "bfloat16",
        batch_size : int = 1,
        temperature : float = 0.0,
        top_p : float = 1.0,
        max_tokens : int = 256,
        num_samples: int = 200,
        num_gpus: int = 1,
        trust_remote_code: bool = True,
    ) -> None:

        super().__init__(model_name)

        print("Initializing a decoder model in transformer: {} ...".format(model_name))
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

        if self.batch_size != 1:
            logger.warning(f"Batch size must be 1 for transformer, but got {self.batch_size}")
            self.batch_size = 1
        if self.temperature == 0:
            if self.num_samples != 1:
                logger.warning(f"Number of samples must be 1 for temperature 0, but got {self.num_samples}")
                self.num_samples = 1

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path = self.model_name,
            torch_dtype = self.dtype,
            trust_remote_code = self.trust_remote_code
        )
        
        # Move model to GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        print(f"Initializing with num_samples: {self.num_samples}")
        self.sampling_params = dict(
            do_sample = self.temperature > 0,
            temperature = self.temperature,
            top_p = self.top_p,
            num_return_sequences = self.num_samples,
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
        self.eos = eos

    def truncate_eos(self, output: str) -> str:
        for stop_word in self.eos:
            if stop_word in output:
                output = output[:output.index(stop_word)]
        return output
        
    def release_memory(self):

        import gc
        import torch

        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    def generate(
            self,
            prompt_set: List[Dict],
            response_prefix: str = "",
            response_suffix: str = ""
        ) -> List[str]:

        logger.info("Example Prompt:\n{}", prompt_set[0]['prompt'])

        generation_set = []

        for prompt in tqdm(prompt_set):
            # Tokenize input
            input_text = make_chat_template(
                prompt = prompt['prompt'],
                response_prefix = "",
                is_chat = self.is_chat(),
                tokenizer = self.tokenizer
            )

            inputs_ids = self.tokenizer(
                input_text,
                return_tensors="pt"
            )

            # Move inputs to GPU if model is on GPU
            if torch.cuda.is_available():
                # Move inputs_ids and attention_mask to GPU
                inputs_ids = {k: v.to(self.model.device) for k, v in inputs_ids.items()}

            input_token_length = inputs_ids['input_ids'].shape[1]

            # shape[0] is the batch size, shape[1] is the sequence length
            self.sampling_params['max_new_tokens'] = self.max_tokens - input_token_length

            outputs = self.model.generate(
                **inputs_ids,
                **self.sampling_params
            )

            # Decode each generated sequence
            for completion_idx, output in enumerate(outputs):
                # Decode the generated tokens (skip the input tokens)
                generated_tokens = output[input_token_length:]
                completion = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                completion = self.truncate_eos(completion)
                
                # Process completion with prefix/suffix
                if self.is_chat():
                    completion = refine_text(completion)
                else:
                    completion = refine_text(prompt['prompt'] + completion)
                
                generation_set.append({
                    'task_id': prompt['task_id'],
                    'completion_id': completion_idx,
                    'completion': completion,
                })

        if len(generation_set) != len(prompt_set) * self.num_samples:
            logger.error("Number of generations does not match the expected number.")

        self.release_memory()

        return generation_set