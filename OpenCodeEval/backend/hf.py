import os
import sys
import fla
import torch
import multiprocessing

from tqdm import tqdm
from loguru import logger
from typing import List, Dict

from OpenCodeEval.backend.base import Generator
from OpenCodeEval.backend.utils import make_chat_template
from OpenCodeEval.utils import refine_text

def stop_at_stop_token_static(text: str, stop_tokens: List[str]) -> str:
    """静态函数：在stop token处停止文本生成"""
    if stop_tokens is None:
        return text
    
    min_stop_index = len(text)
    for stop_token in stop_tokens:
        stop_index = text.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return text[:min_stop_index]

class TransformerGenerator(Generator):

    def __init__(
        self,
        model_name: str,
        tokenizer_name: str = None,
        model_type: str = "Base",
        dtype: str = "bfloat16",
        response_prefix: str = "",
        response_suffix: str = "",
        batch_size : int = 1,
        temperature : float = 0.0,
        top_p : float = 1.0,
        max_tokens : int = 256,
        num_samples: int = 200,
        num_gpus: int = 1,
        trust_remote_code: bool = True,
        seed: int = 0
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
        self.seed = seed

        logger.info(f"Initializing with seed: {seed}")

        if self.batch_size != 1:
            logger.warning(f"Batch size must be 1 for transformer, but got {self.batch_size}")
            self.batch_size = 1
        if self.temperature == 0:
            if self.num_samples != 1:
                logger.warning(f"Number of samples must be 1 for temperature 0, but got {self.num_samples}")
                self.num_samples = 1

        # 检查可用的GPU数量
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            if self.num_gpus > available_gpus:
                logger.warning(f"Requested {self.num_gpus} GPUs but only {available_gpus} available. Using {available_gpus} GPUs.")
                self.num_gpus = available_gpus
        else:
            logger.warning("CUDA not available, falling back to CPU")
            self.num_gpus = 0

        print(f"Initializing with num_samples: {self.num_samples}, num_gpus: {self.num_gpus}")
        self.sampling_params = dict(
            do_sample = self.temperature > 0,
            top_p = self.top_p,
            num_return_sequences = self.num_samples,
            top_k = 0
        )
        
        # 只有当温度大于0时才添加温度参数
        if self.temperature > 0:
            self.sampling_params['temperature'] = self.temperature
    
    def is_chat(self) -> bool:
        if self.model_type == "Chat":
            if self.tokenizer.chat_template is None:
                logger.error("When the model type is Chat, the chat template must be set for the tokenizer.")
            else:
                return True
        else:
            return False
        
    def set_stop(self, eos: List[str]):
        self.stop_tokens = eos
    

        
    def release_memory(self):
        import gc
        import torch

        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    def _process_batch_on_gpu(
        self,
        gpu_id: int,
        prompt_batch: List[Dict],
        model_name: str,
        model_type: str,
        dtype: str,
        trust_remote_code: bool,
        sampling_params: dict,
        max_tokens: int,
        stop_tokens: List[str],
        seed: int,
        response_prefix: str = "",
        response_suffix: str = ""
    ) -> List[Dict]:
        """在指定GPU上处理一批prompt"""
        # 设置随机种子确保结果一致性
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # 设置确定性模式
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # 设置更多确定性选项
            torch.use_deterministic_algorithms(True)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # 设置当前进程使用的GPU
        if torch.cuda.is_available():
            # 确保CUDA已经初始化
            torch.cuda.init()
            torch.cuda.set_device(gpu_id)
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # 在每个GPU上加载模型和tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            # 确保模型加载的确定性
            low_cpu_mem_usage=False,
            device_map=None
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        # 确保模型处于评估模式
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # 检查是否为chat模型
        def is_chat():
            if model_type == "Chat":
                if tokenizer.chat_template is None:
                    logger.error("When the model type is Chat, the chat template must be set for the tokenizer.")
                else:
                    return True
            else:
                return False

        if gpu_id == 0:
            logger.info(f"Sampling params: {sampling_params}")
            logger.info(f"Stop tokens: {stop_tokens}")
            logger.info(f"Model dtype: {model.dtype}")
            logger.info(f"Model device: {next(model.parameters()).device}")
            logger.info(f"Model mode: {model.training}")

        generation_set = []
        
        for prompt in tqdm(prompt_batch, desc=f"GPU {gpu_id} Processing", leave=False):
            # Tokenize input
            input_text = make_chat_template(
                prompt=prompt['prompt'],
                response_prefix=response_prefix,
                is_chat=is_chat(),
                tokenizer=tokenizer
            )

            inputs_tokens = tokenizer(
                input_text,
                return_tensors="pt"
            )
            # logger.info(f"Inputs Tokens:\n{inputs_tokens}")

            # Move inputs to GPU if model is on GPU
            if torch.cuda.is_available():
                inputs_tokens = {k: v.to(model.device) for k, v in inputs_tokens.items()}

            # Generate
            sampling_params = sampling_params.copy()
            sampling_params['max_new_tokens'] = max_tokens

            outputs = model.generate(
                **inputs_tokens,
                **sampling_params
            )

            # Decode each generated sequence
            for completion_idx, output in enumerate(outputs):
                completion = tokenizer.decode(output, skip_special_tokens=False)
                logger.info(f"Original Completion:\n{completion}")
                completion = completion.split(input_text)[-1]
                logger.info(f"After Prompt Completion:\n{completion}")
                # 处理stop tokens
                completion = stop_at_stop_token_static(completion, stop_tokens)
                logger.info(f"After Stop Tokens Completion:\n{completion}")
                completion = refine_text(completion)
                logger.info(f"After Refine Completion:\n{completion}")
                if not is_chat():
                    completion = input_text + completion
                logger.info(f"Saved Completion:\n{completion}")
                
                generation_set.append({
                    'task_id': prompt['task_id'],
                    'completion_id': completion_idx,
                    'completion': completion,
                })
        
        # 清理GPU内存
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        return generation_set

    def generate(
            self,
            prompt_set: List[Dict],
            response_prefix: str = "",
            response_suffix: str = ""
        ) -> List[str]:

        logger.info(f"Example Prompt:\n{prompt_set[0]['prompt']}")
        logger.info(f"Processing {len(prompt_set)} prompts on {self.num_gpus} GPUs")

        if self.num_gpus <= 1:
            # 单GPU或CPU处理
            return self._process_batch_on_gpu(
                0,
                prompt_set,
                self.model_name,
                self.model_type,
                self.dtype,
                self.trust_remote_code,
                self.sampling_params,
                self.max_tokens,
                getattr(self, 'stop_tokens', []),
                self.seed, 
                response_prefix,
                response_suffix
            )
        
        # 多GPU并行处理
        # 将prompt_set分成多个批次，每个GPU处理一部分
        batch_size = len(prompt_set) // self.num_gpus
        if len(prompt_set) % self.num_gpus != 0:
            batch_size += 1
        
        prompt_batches = []
        for i in range(0, len(prompt_set), batch_size):
            prompt_batches.append(prompt_set[i:i + batch_size])
        
        # 确保批次数量不超过GPU数量
        while len(prompt_batches) < self.num_gpus:
            prompt_batches.append([])
        
        # 使用多进程在多个GPU上并行处理
        # 临时设置spawn启动方法，仅用于这个进程池
        original_start_method = multiprocessing.get_start_method()
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
            
        with multiprocessing.Pool(processes=self.num_gpus) as pool:
            # 并行处理每个批次
            results = []
            for gpu_id, prompt_batch in enumerate(prompt_batches):
                if prompt_batch:  # 只处理非空批次
                    result = pool.apply_async(
                        self._process_batch_on_gpu, 
                        args=(
                            gpu_id,
                            prompt_batch,
                            self.model_name,
                            self.model_type,
                            self.dtype,
                            self.trust_remote_code,
                            self.sampling_params,
                            self.max_tokens,
                            getattr(self, 'stop_tokens', []),
                            self.seed,
                            response_prefix,
                            response_suffix)
                    )
                    results.append((gpu_id, result))
            
            # 按GPU ID顺序收集所有结果，确保顺序一致
            generation_set = []
            logger.info("Collecting results from all GPUs...")
            for gpu_id, result in tqdm(sorted(results, key=lambda x: x[0]), desc="GPU Processing", total=len(results)):
                generation_set.extend(result.get())
        
        # 恢复原来的multiprocessing启动方法
        try:
            multiprocessing.set_start_method(original_start_method, force=True)
        except RuntimeError:
            pass
        
        if len(generation_set) != len(prompt_set) * self.num_samples:
            logger.error("Number of generations does not match the expected number.")

        return generation_set