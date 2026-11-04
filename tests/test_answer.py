import json

import httpx
import pytest

from askrepo.answer import AnswerError, answer_question, chat_completions_url, complete_chat
from askrepo.config import Settings
from askrepo.types import Chunk, Hit


def test_chat_url_joins_the_api_root() -> None:
    assert chat_completions_url("https://example.test/v1") == "https://example.test/v1/chat/completions"
    assert chat_completions_url("https://example.test/v1/") == "https://example.test/v1/chat/completions"
    assert (
        chat_completions_url("https://example.test/v1/chat/completions")
        == "https://example.test/v1/chat/completions"
    )


def test_settings_treat_blank_key_as_offline() -> None:
    blank = Settings.from_env(
        {
            "ASKREPO_API_KEY": "  ",
            "ASKREPO_BASE_URL": "http://localhost:8000/v1",
            "ASKREPO_MODEL": "local-model",
            "ASKREPO_RETRIEVER": "lexical",
        }
    )
    assert blank.api_key is None
    assert blank.base_url == "http://localhost:8000/v1"
    assert blank.model == "local-model"

    defaults = Settings.from_env({})
    assert defaults.model == "gpt-4o-mini"
    assert defaults.base_url == "https://api.openai.com/v1"
    assert defaults.retriever == "lexical"


def test_offline_answer_does_not_touch_the_network() -> None:
    result = answer_question(
        "where is the invoice total calculated",
        [_hit()],
        _settings(api_key=None),
        transport=_explode(),
    )

    assert result.mode == "offline"
    assert result.model is None
    assert "src/billing/invoice.py" in result.text
    assert "ASKREPO_API_KEY" in result.text


def test_empty_hits_skip_the_model_even_with_a_key() -> None:
    result = answer_question(
        "invoice",
        [],
        _settings(api_key="test-key"),
        transport=_explode(),
    )

    assert result.mode == "offline"
    assert "not called" in result.text


def test_model_request_uses_key_model_and_excerpts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content.decode())
        assert body["model"] == "demo-model"
        assert request.url.path.endswith("/chat/completions")
        user = body["messages"][1]["content"]
        assert "invoice total" in user
        assert "src/billing/invoice.py" in user
        assert "calculate_invoice_total" in user
        return httpx.Response(200, json={"choices": [{"message": {"content": "See invoice.py."}}]})

    result = answer_question(
        "where is the invoice total calculated",
        [_hit()],
        _settings(api_key="test-key", model="demo-model"),
        transport=httpx.MockTransport(handler),
    )

    assert result.mode == "model"
    assert result.model == "demo-model"
    assert result.text == "See invoice.py."


def test_http_error_is_an_answer_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="nope")

    with pytest.raises(AnswerError, match="chat API request failed"):
        complete_chat(
            _settings(api_key="test-key"),
            [{"role": "user", "content": "hi"}],
            transport=httpx.MockTransport(handler),
        )


def _hit() -> Hit:
    chunk = Chunk(
        id="src/billing/invoice.py:1-10",
        path="src/billing/invoice.py",
        start_line=1,
        end_line=10,
        text="def calculate_invoice_total(lines):\n    return sum(lines)\n",
        language="python",
        symbols=("calculate_invoice_total",),
        imports=("billing.tax",),
    )
    return Hit(chunk=chunk, score=1.5, reasons=("symbol:calculate_invoice_total",))


def _settings(api_key: str | None, model: str = "demo-model") -> Settings:
    return Settings(
        api_key=api_key,
        base_url="http://example.test/v1",
        model=model,
        retriever="lexical",
    )


def _explode() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request to {request.url}")

    return httpx.MockTransport(handler)
