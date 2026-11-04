"""Environment-backed settings. Nothing here reads the index or the network."""

import os
from dataclasses import dataclass
from typing import Mapping


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_RETRIEVER = "lexical"


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str
    model: str
    retriever: str

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ
        raw_key = env.get("ASKREPO_API_KEY", "").strip()
        base_url = env.get("ASKREPO_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
        model = env.get("ASKREPO_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        retriever = env.get("ASKREPO_RETRIEVER", DEFAULT_RETRIEVER).strip() or DEFAULT_RETRIEVER
        return cls(
            api_key=raw_key or None,
            base_url=base_url,
            model=model,
            retriever=retriever,
        )
