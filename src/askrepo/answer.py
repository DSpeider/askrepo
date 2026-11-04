"""Turn retrieved chunks into an answer.

With ``ASKREPO_API_KEY`` set, the excerpts are sent to an OpenAI-compatible
``/chat/completions`` endpoint. Without a key, or when retrieval returns
nothing, this module stays offline and does not open a socket.

Ranking does not happen here. A new retriever (embeddings, a code graph) only
has to return ``Hit`` values.
"""

from __future__ import annotations

import json

import httpx

from askrepo.config import Settings
from askrepo.types import Answer, Hit


SYSTEM_PROMPT = (
    "You answer questions about a local code repository. "
    "Use only the retrieved excerpts in the user message. "
    "Cite paths with line ranges when you rely on them. "
    "If the excerpts do not contain the answer, say what is missing. "
    "Do not invent files, symbols, or behavior."
)

_EXCERPT_CHARS = 3_500


class AnswerError(RuntimeError):
    """The chat API was called and did not return usable text."""


def answer_question(
    question: str,
    hits: list[Hit],
    settings: Settings,
    *,
    transport: httpx.BaseTransport | None = None,
) -> Answer:
    """Answer ``question`` from ``hits``.

    ``transport`` is for tests. The CLI leaves it unset, which uses the network
    only when a key is present and at least one hit was retrieved.
    """

    if not hits:
        return Answer(text=_empty_text(), mode="offline", model=None)
    if not settings.api_key:
        return Answer(text=_offline_text(hits), mode="offline", model=None)
    text = complete_chat(settings, build_messages(question, hits), transport)
    return Answer(text=text, mode="model", model=settings.model)


def build_messages(question: str, hits: list[Hit]) -> list[dict[str, str]]:
    blocks: list[str] = []
    for hit in hits:
        chunk = hit.chunk
        text = chunk.text
        if len(text) > _EXCERPT_CHARS:
            text = text[:_EXCERPT_CHARS] + "\n…"
        symbols = ", ".join(chunk.symbols) if chunk.symbols else "-"
        imports = ", ".join(chunk.imports) if chunk.imports else "-"
        blocks.append(
            "\n".join(
                [
                    "---",
                    f"path: {chunk.path}",
                    f"lines: {chunk.start_line}-{chunk.end_line}",
                    f"language: {chunk.language}",
                    f"symbols: {symbols}",
                    f"imports: {imports}",
                    "text:",
                    text.rstrip(),
                ]
            )
        )
    user = "Question:\n" + question.strip() + "\n\nExcerpts:\n" + "\n".join(blocks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def chat_completions_url(base_url: str) -> str:
    """Join ``ASKREPO_BASE_URL`` with the chat path.

    The base URL is the API root, usually ending in ``/v1``. A base that
    already ends in ``/chat/completions`` is left alone.
    """

    root = base_url.strip().rstrip("/")
    suffix = "/chat/completions"
    if root.endswith(suffix):
        return root
    return root + suffix


def complete_chat(
    settings: Settings,
    messages: list[dict[str, str]],
    transport: httpx.BaseTransport | None = None,
) -> str:
    if not settings.api_key:
        raise AnswerError("ASKREPO_API_KEY is not set")
    try:
        with httpx.Client(timeout=60.0, transport=transport) as client:
            response = client.post(
                chat_completions_url(settings.base_url),
                headers={"Authorization": f"Bearer {settings.api_key}"},
                json={
                    "model": settings.model,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 800,
                },
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise AnswerError(f"chat API request failed: {exc}") from exc
    try:
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise AnswerError("chat API returned an unexpected payload") from exc
    if not isinstance(content, str) or not content.strip():
        raise AnswerError("chat API returned an empty message")
    return content.strip()


def _offline_text(hits: list[Hit]) -> str:
    seen: set[str] = set()
    files: list[str] = []
    for hit in hits:
        path = hit.chunk.path
        if path in seen:
            continue
        seen.add(path)
        files.append(f"- {path}:{hit.chunk.start_line}-{hit.chunk.end_line}")
    return (
        "No ASKREPO_API_KEY is set, so no model was called.\n"
        "The retrieved excerpts are the context that would be sent to an "
        "OpenAI-compatible chat API.\n"
        "Set ASKREPO_API_KEY to synthesize an answer. "
        "Optional: ASKREPO_BASE_URL (API root, usually ending in /v1) and ASKREPO_MODEL.\n"
        "\n"
        "Strongest files:\n" + "\n".join(files)
    )


def _empty_text() -> str:
    return (
        "No chunks scored above zero, so a model was not called.\n"
        "Ask with a file name, function name, or a distinctive string from the code, "
        "and re-run `askrepo index` if the tree changed."
    )
