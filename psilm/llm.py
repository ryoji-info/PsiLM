"""Language hemisphere: a thin mlx-lm wrapper with deterministic decoding."""

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


class LLM:
    def __init__(self, model_id: str = DEFAULT_MODEL):
        self.model_id = model_id
        self.model, self.tokenizer = load(model_id)
        self.sampler = make_sampler(temp=0.0)

    def chat(self, system: str, user: str, max_tokens: int = 400) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=self.sampler,
            verbose=False,
        )
