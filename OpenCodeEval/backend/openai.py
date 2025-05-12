import os

from typing import List, Dict
from openai import OpenAI
from loguru import logger
from tqdm.contrib.concurrent import thread_map

from OpenCodeEval.backend.base import Generator

class OpenaiGenerator(Generator):

    def __init__(
            self,
            model_name: str,
            model_type: str = "Instruction",
            batch_size : int = 1,
            temperature : float = 0.0,
            top_p : float = 1.0,
            max_tokens : int = 1024,
            num_samples : int = 1,
        ) -> None:

        super().__init__(model_name)

        print("Initializing a decoder model in openai: {} ...".format(model_name))
        self.model_type = model_type
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.num_samples = num_samples

        self.client = OpenAI(
            base_url = os.getenv("OPENAI_BASE_URL"),
            api_key = os.getenv("OPENAI_API_KEY")
        )

    def is_chat(self) -> bool:
        if self.model_type == "Chat":
            return True
        else:
            return False
    
    def set_stop(self, eos: List[str]):
        self.eos = eos

    def connect_server(self, prompt):

        num_tries = 5

        try:

            for _ in range(num_tries):

                result = self.client.chat.completions.create(
                    model = self.model_name,
                    messages=[
                        {"role": "user", "content": prompt['prompt']}
                    ], 
                    n = self.num_samples,
                    stream = False,
                    # stop = self.eos,
                    temperature = self.temperature,
                    # top_p = self.top_p,
                    max_tokens = self.max_tokens,
                    extra_headers = {'apikey':os.getenv("OPENAI_API_KEY")},
                )

                if all(choice.message.content for choice in result.choices):
                    break
                else:
                    logger.warning("No content in the response, retrying...")
                
            results = [
                dict(
                    task_id=prompt['task_id'],
                    completion_id=i,
                    completion=choice.message.content
                )
                for i, choice in enumerate(result.choices)
            ]
            
        except Exception as e:
            logger.error("Error: {}".format(e))
            results = [
                dict(
                    task_id=prompt['task_id'],
                    completion_id=i,
                    completion = ''
                )
                for i in range(self.num_samples)
            ]
            
        return results

    def generate(
        self,
        prompt_set: List[Dict],
        response_prefix: str = "",
        response_suffix: str = ""
    ) -> List[Dict]:
        
        logger.info("Example Prompt:\n{}", prompt_set[0]['prompt'])
        
        results = thread_map(
            self.connect_server, 
            prompt_set,
            max_workers = self.batch_size,
            desc="Generating Completions"
        )
        

        generations = [item for sublist in results for item in sublist]
        
        return generations