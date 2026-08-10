"""Open-weights instruct LLM — backs `llm_query` (OK-VQA) and `select_answer` (NExT-QA).

# DEVIATION D2: paper uses GPT-3 `text-davinci-003` (deprecated). We use an
# open-weights instruct model (Llama-3.1-8B-Instruct by default). See docs/deviations.md.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class MockTextLLM:
    def llm_query(self, question: str, long_answer: bool = True) -> str:
        return "mock-llm-answer"

    def select_answer(self, info: str, question: str, options) -> int:
        return 0


class TextLLM:
    def __init__(self, cfg):
        self.cfg = cfg
        c = cfg.llm_qa
        self.model_name = c.model
        self.temperature = float(c.temperature)
        self._model = None
        self._tokenizer = None

    def _lazy_load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading text LLM: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )

    def _chat(self, user_msg: str, max_new_tokens: int = 64) -> str:
        self._lazy_load()
        messages = [{"role": "user", "content": user_msg}]
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        gen = self._model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None,
        )
        return self._tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def llm_query(self, question: str, long_answer: bool = True) -> str:
        suffix = "" if long_answer else " Answer in one or two words."
        return self._chat(question + suffix, max_new_tokens=200 if long_answer else 16)

    def select_answer(self, info: str, question: str, options) -> int:
        opts = "\n".join(f"{i}. {o}" for i, o in enumerate(options))
        prompt = (
            f"Context: {info}\nQuestion: {question}\nOptions:\n{opts}\n"
            "Reply with only the number of the best option."
        )
        out = self._chat(prompt, max_new_tokens=8)
        import re

        m = re.search(r"\d+", out)
        idx = int(m.group()) if m else 0
        return idx if 0 <= idx < len(options) else 0


def build_text_llm(cfg, mock: bool = False):
    return MockTextLLM() if mock else TextLLM(cfg)
