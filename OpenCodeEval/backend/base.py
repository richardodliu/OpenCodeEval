from typing import Callable
from abc import ABC, abstractmethod

def make_chat_template(
        prompt: str,
        response_prefix: str = "",
        is_chat: bool = True,
        tokenizer: Callable = None
    ) -> str:

    if is_chat:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content":  prompt},
            ],
            tokenize = False,
            add_generation_prompt = True
        ) + response_prefix
        if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
            prompt = prompt[len(tokenizer.bos_token):]
        return prompt
    else:
        return prompt

class Generator(ABC):

    model_name: str = None

    def __init__(self, model_name: str) -> None:
        """
        :param stop_words: list
            list of stop words if the generation uses a stopping criteria during generation
        :param requires_execution: bool
            wheter the task requires code execution during evaluation or not
        """
        self.model_name = model_name

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    @abstractmethod
    def set_stop(self):
        """
        Set the stop tokens for the model
        """
        pass

    @abstractmethod
    def generate(self):
        """Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass