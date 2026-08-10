"""Program generator π — open-weights code model (Qwen2.5-Coder), replaces Codex.

# DEVIATION D1: paper uses OpenAI Codex `code-davinci-002` (deprecated 2023-03-23).
# We use an open-weights code model. See docs/deviations.md.

Greedy decoding (temperature 0) to mirror the paper's deterministic setting.
The `MockCodeGen` variant returns a fixed program so the smoke path / tests run
with no GPU or weights.
"""
from __future__ import annotations

import logging
import re

from vipergpt_repro.pipeline.prompt import build_prompt

logger = logging.getLogger(__name__)

_CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_program(text: str) -> str:
    """Pull the execute_command function out of a model completion."""
    m = _CODE_BLOCK.search(text)
    if m:
        text = m.group(1)
    # Keep from the first 'def execute_command' onward.
    idx = text.find("def execute_command")
    if idx != -1:
        text = text[idx:]
    return text.strip()


class MockCodeGen:
    """Deterministic generator for smoke/tests. NOT valid for reported numbers."""

    def generate(self, query: str) -> str:
        return (
            "def execute_command(image):\n"
            "    image_patch = ImagePatch(image)\n"
            "    return image_patch.simple_query('What is this?')\n"
        )


class CodeGen:
    def __init__(self, cfg):
        self.cfg = cfg
        c = cfg.codegen
        self.model_name = c.model
        self.temperature = float(c.temperature)
        self.max_new_tokens = int(c.max_new_tokens)
        self.prompt_path = c.prompt
        self._model = None
        self._tokenizer = None

    def _lazy_load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading code generator: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dtype = getattr(torch, str(self.cfg.codegen.get("dtype", "bfloat16")))
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map="auto"
        )

    def generate(self, query: str) -> str:
        self._lazy_load()
        prompt = build_prompt(query, prompt_path=self.prompt_path)
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        gen = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None,
        )
        completion = self._tokenizer.decode(
            gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return extract_program(completion)


def build_codegen(cfg, mock: bool = False):
    return MockCodeGen() if mock else CodeGen(cfg)
