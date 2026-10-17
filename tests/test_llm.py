"""LLM client: configuration, retries with backoff, friendly errors, tool-call parsing."""

from __future__ import annotations

from types import SimpleNamespace

import openai
import pytest

try:  # openai >= 3 ships its HTTP layer as httpx2; older releases use httpx
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

from agent.llm import (
    PROVIDERS,
    LLMError,
    LLMSettings,
    MockLLM,
    OpenAICompatibleLLM,
    check_connection,
    infer_provider,
    settings_for_provider,
    settings_from_env,
    text_reply,
    tool_reply,
)

REQUEST = httpx.Request("POST", "https://api.example.com/v1/chat/completions")


def _status_error(cls, status: int, headers: dict | None = None):
    return cls("boom", response=httpx.Response(status, request=REQUEST, headers=headers or {}), body=None)


def _completion(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool(name: str, arguments: str, call_id: str = "call_1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


SETTINGS = LLMSettings(provider="openai", base_url="https://api.openai.com/v1", api_key="sk-test", model="m")


def make(outcomes, **kwargs):
    sleeps: list[float] = []
    client = FakeClient(outcomes)
    llm = OpenAICompatibleLLM(SETTINGS, client=client, sleep=sleeps.append, base_delay=0.5, **kwargs)
    return llm, client, sleeps


# --- settings ---------------------------------------------------------------------


def test_env_settings_with_provider_preset():
    s = settings_from_env(env={"LLM_PROVIDER": "groq", "LLM_API_KEY": "gsk"})
    assert s.base_url == PROVIDERS["groq"].base_url
    assert s.model == PROVIDERS["groq"].default_model
    assert s.configured


def test_env_wins_over_secrets_and_provider_is_inferred():
    secrets = {"LLM_BASE_URL": "https://openrouter.ai/api/v1", "LLM_API_KEY": "from-secrets", "LLM_MODEL": "x/y"}
    s = settings_from_env(secrets=secrets, env={"LLM_API_KEY": "from-env"})
    assert s.provider == "openrouter"
    assert s.api_key == "from-env"
    assert s.model == "x/y"


def test_secrets_that_raise_are_ignored():
    class Broken:
        def get(self, key):
            raise FileNotFoundError("no secrets.toml")

    s = settings_from_env(secrets=Broken(), env={})
    assert not s.configured
    assert s.missing() == ["LLM_API_KEY"]


def test_ollama_needs_no_key():
    s = settings_for_provider("ollama")
    assert s.configured and not s.needs_key
    assert infer_provider("http://localhost:11434/v1") == "ollama"
    custom_local = settings_for_provider("custom", base_url="http://127.0.0.1:8000/v1", model="local")
    assert custom_local.configured


def test_key_never_appears_in_repr():
    assert "sk-test" not in repr(SETTINGS)


def test_unconfigured_client_raises_friendly_error():
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        OpenAICompatibleLLM(LLMSettings(provider="openai", base_url="https://x", model="m"))


# --- calls --------------------------------------------------------------------------


def test_tool_calls_are_parsed_and_tools_sent():
    llm, client, _ = make([_completion(None, [_tool("run_python", '{"code": "result = 1"}')])])
    reply = llm.complete([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    assert reply.tool_calls[0].name == "run_python"
    assert reply.tool_calls[0].arguments == {"code": "result = 1"}
    assert client.requests[0]["tool_choice"] == "auto"
    assert "temperature" not in client.requests[0]  # reasoning models reject it
    message = reply.to_message()
    assert message["tool_calls"][0]["function"]["arguments"] == '{"code": "result = 1"}'


def test_invalid_tool_arguments_are_flagged_not_raised():
    llm, _, _ = make([_completion(None, [_tool("run_python", "{not json")])])
    call = llm.complete([]).tool_calls[0]
    assert call.parse_error and call.arguments == {}


def test_think_blocks_are_stripped():
    llm, _, _ = make([_completion("<think>hmm</think>\nready")])
    assert llm.complete([]).content == "ready"


def test_retries_with_exponential_backoff_then_succeeds():
    llm, client, sleeps = make(
        [
            _status_error(openai.RateLimitError, 429),
            _status_error(openai.InternalServerError, 503),
            openai.APITimeoutError(request=REQUEST),
            _completion("ok"),
        ]
    )
    assert llm.complete([]).content == "ok"
    assert len(client.requests) == 4
    assert len(sleeps) == 3
    assert sleeps[0] < sleeps[1] < sleeps[2]  # 0.5, 1, 2 (+ small jitter)


def test_retry_after_header_is_respected():
    llm, _, sleeps = make([_status_error(openai.RateLimitError, 429, {"retry-after": "3"}), _completion("ok")])
    llm.complete([])
    assert sleeps == [3.0]


def test_gives_up_after_max_retries():
    llm, client, sleeps = make([_status_error(openai.RateLimitError, 429)] * 3, max_retries=2)
    with pytest.raises(LLMError, match="rate-limiting"):
        llm.complete([])
    assert len(client.requests) == 3 and len(sleeps) == 2


@pytest.mark.parametrize(
    ("exc", "text"),
    [
        (_status_error(openai.AuthenticationError, 401), "rejected the API key"),
        (_status_error(openai.NotFoundError, 404), "was not found"),
        (_status_error(openai.PermissionDeniedError, 403), "refused access"),
        (openai.APIConnectionError(request=REQUEST), None),
    ],
)
def test_errors_become_friendly_messages(exc, text):
    if isinstance(exc, openai.APIConnectionError):
        llm, _, _ = make([exc] * 5)
        text = "Couldn't reach"
    else:
        llm, client, sleeps = make([exc])
    with pytest.raises(LLMError, match=text):
        llm.complete([])


def test_bad_request_about_tools_explains_model_choice():
    exc = openai.BadRequestError("tools are not supported", response=httpx.Response(400, request=REQUEST), body=None)
    llm, _, _ = make([exc])
    with pytest.raises(LLMError, match="doesn't support tool calling"):
        llm.complete([])


def test_empty_response_is_an_error():
    llm, _, _ = make([SimpleNamespace(choices=[])])
    with pytest.raises(LLMError, match="empty response"):
        llm.complete([])


# --- mock -------------------------------------------------------------------------


def test_mock_records_payloads_and_replays_script():
    mock = MockLLM([tool_reply("run_python", code="result = 1"), text_reply("done")])
    first = mock.complete([{"role": "user", "content": "q"}], tools=[{"t": 1}])
    assert first.tool_calls[0].arguments == {"code": "result = 1"}
    assert mock.complete([]).content == "done"
    assert len(mock.calls) == 2 and mock.calls[0]["tools"] == [{"t": 1}]
    with pytest.raises(LLMError):
        mock.complete([])


def test_check_connection():
    assert check_connection(MockLLM([text_reply("ready")])) == "ready"
