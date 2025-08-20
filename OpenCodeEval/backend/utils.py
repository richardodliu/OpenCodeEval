from typing import Callable

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