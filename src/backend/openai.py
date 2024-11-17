import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT))])

from tqdm import tqdm
from typing import List
from openai import OpenAI
from typing import Dict

from backend.base import Generator

from utils import refine_text, multi_process_function, program_extract

class OpenaiGenerator(Generator):
    def __init__(self,
                 model_name: str,
                 model_type: str = "Instruction",
                 batch_size : int = 1,
                 temperature : float = 0.0,
                 max_tokens : int = 1024,
                 num_samples : int = 200,
                 eos: List[str] = None
                ) -> None:
        super().__init__(model_name)

        print("Initializing a decoder model: {} ...".format(model_name))
        self.model_type = model_type
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_samples = num_samples
        self.eos = eos

        self.client = OpenAI(
            base_url = 'http://openai.infly.tech/v1/',
            api_key='sk-z4S1UssPDKChWJgoc6EMRorXmB25kXFkgGkpZfXgxpfBzZkk'
        )

    def is_chat(self) -> bool:
        if self.model_type == "Chat":
            return True
        else:
            return False

    def connect_server(self, prompt, num_samples):
        try:
            result = self.client.chat.completions.create(
                model = self.model_name,
                messages=[
                    {"role": "user", "content": prompt['prompt']}
                ], 
                n = num_samples,
                stream = False,
                temperature = self.temperature,
                max_tokens = self.max_tokens,
                stop = self.eos,
                extra_headers={'apikey': 'sk-z4S1UssPDKChWJgoc6EMRorXmB25kXFkgGkpZfXgxpfBzZkk'},
            )
            # 为每个回复创建一个字典，并收集到列表中
            results = [
                dict(
                    task_id=prompt['task_id'],
                    completion_id=i,
                    completion=choice.message.content
                )
                for i, choice in enumerate(result.choices)
            ]
            
        except Exception as e:
            # 发生错误时，创建相同数量的错误信息字典
            results = [
                dict(
                    task_id=prompt['task_id'],
                    completion_id=i,
                    completion='error:{}'.format(e)
                )
                for i in range(num_samples)
            ]
            
        return results
    
    def generate(self,
                 prompt_set: List[Dict],
                 eos: List[str] = None,
                 response_prefix: str = "",
                 response_suffix: str = ""
                ) -> List[Dict]:  # 修改返回类型注释
        
        results = multi_process_function(self.connect_server, 
                                        prompt_set[:10],
                                        num_workers = self.batch_size,
                                        desc="Generating")
        
        # 展平结果列表（因为每个 prompt 会返回多个结果）
        generations = [item for sublist in results for item in sublist]
        
        return generations
        
        
        
        
        

        