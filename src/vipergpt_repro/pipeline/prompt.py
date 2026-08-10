"""Prompt assembly: inject the query into the API-spec prompt template."""
from __future__ import annotations

from pathlib import Path

from vipergpt_repro.utils.paths import REPO_ROOT

PLACEHOLDER = "INSERT_QUERY_HERE"


def load_prompt_template(prompt_path: str | Path | None = None) -> str:
    path = Path(prompt_path) if prompt_path else REPO_ROOT / "prompts" / "api.prompt"
    return path.read_text(encoding="utf-8")


def build_prompt(query: str, template: str | None = None, prompt_path: str | Path | None = None) -> str:
    template = template if template is not None else load_prompt_template(prompt_path)
    if PLACEHOLDER not in template:
        raise ValueError(f"Prompt template missing placeholder {PLACEHOLDER!r}")
    return template.replace(PLACEHOLDER, query.strip())
