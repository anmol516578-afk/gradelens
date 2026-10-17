"""Provider-agnostic LLM client (any OpenAI-compatible endpoint) and a MockLLM for tests.

Configuration comes from environment variables or Streamlit secrets:

    LLM_PROVIDER   openai | gemini | groq | openrouter | ollama | custom   (optional)
    LLM_BASE_URL   e.g. https://api.openai.com/v1                           (optional with a provider)
    LLM_API_KEY    the provider's API key                                   (not needed for Ollama)
    LLM_MODEL      e.g. gpt-5.4-mini                                        (optional with a provider)

Every model is reached through the OpenAI Python SDK's Chat Completions API
with tool (function) calling.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import re
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

log = logging.getLogger("gradelens.llm")


# --- Providers & settings --------------------------------------------------------


@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    base_url: str
    default_model: str
    key_url: str | None = None
    needs_key: bool = True


# Default models checked against each provider's documentation in October 2026.
# Any model with tool calling works; change LLM_MODEL to switch.
PROVIDERS: dict[str, Provider] = {
    p.key: p
    for p in (
        Provider("openai", "OpenAI", "https://api.openai.com/v1", "gpt-5.4-mini",
                 "https://platform.openai.com/api-keys"),
        Provider("gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai/",
                 "gemini-3.8-flash", "https://aistudio.google.com/apikey"),
        Provider("groq", "Groq", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b",
                 "https://console.groq.com/keys"),
        Provider("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "openai/gpt-5.4-mini",
                 "https://openrouter.ai/keys"),
        Provider("ollama", "Ollama (local, free)", "http://localhost:11434/v1", "qwen3:8b", None, needs_key=False),
        Provider("custom", "Custom (OpenAI-compatible)", "", "", None),
    )
}  # fmt: skip


def infer_provider(base_url: str) -> str:
    url = (base_url or "").lower()
    for key, hint in (("openai", "api.openai.com"), ("gemini", "generativelanguage.googleapis.com"),
                      ("groq", "groq.com"), ("openrouter", "openrouter.ai"),
                      ("ollama", ":11434")):  # fmt: skip
        if hint in url:
            return key
    return "custom"


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "openai"
    base_url: str = ""
    api_key: str = field(default="", repr=False)
    model: str = ""
    timeout: float = 60.0

    @property
    def provider_info(self) -> Provider:
        return PROVIDERS.get(self.provider, PROVIDERS["custom"])

    @property
    def needs_key(self) -> bool:
        if self.provider == "ollama":
            return False
        return not any(h in self.base_url for h in ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"))

    def missing(self) -> list[str]:
        problems = []
        if not self.base_url:
            problems.append("LLM_BASE_URL")
        if not self.model:
            problems.append("LLM_MODEL")
        if self.needs_key and not self.api_key:
            problems.append("LLM_API_KEY")
        return problems

    @property
    def configured(self) -> bool:
        return not self.missing()

    def describe(self) -> str:
        return f"{self.provider_info.label} · {self.model or '(no model)'}"


def _lookup(name: str, secrets: Mapping[str, Any] | None, env: Mapping[str, str]) -> str:
    # An empty variable counts as unset (docker-compose passes unset keys as ""), so secrets still apply.
    value = (env.get(name) or "").strip() or None
    if value is None and secrets is not None:
        try:
            value = secrets.get(name)
        except Exception:  # noqa: BLE001 - st.secrets raises if no secrets file exists
            value = None
    return str(value).strip() if value is not None else ""


def settings_from_env(secrets: Mapping[str, Any] | None = None, env: Mapping[str, str] | None = None) -> LLMSettings:
    """Environment variables win over Streamlit secrets; provider presets fill the gaps."""
    env = os.environ if env is None else env
    provider = _lookup("LLM_PROVIDER", secrets, env).lower()
    base_url = _lookup("LLM_BASE_URL", secrets, env)
    api_key = _lookup("LLM_API_KEY", secrets, env)
    model = _lookup("LLM_MODEL", secrets, env)
    if provider not in PROVIDERS:
        provider = infer_provider(base_url) if base_url else "openai"
    preset = PROVIDERS[provider]
    return LLMSettings(
        provider=provider,
        base_url=base_url or preset.base_url,
        api_key=api_key,
        model=model or preset.default_model,
    )


def settings_for_provider(provider: str, api_key: str = "", model: str = "", base_url: str = "") -> LLMSettings:
    preset = PROVIDERS.get(provider, PROVIDERS["custom"])
    return LLMSettings(
        provider=preset.key,
        base_url=base_url.strip() or preset.base_url,
        api_key=api_key.strip(),
        model=model.strip() or preset.default_model,
    )


# --- Messages ----------------------------------------------------------------------


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw_arguments: str = ""
    parse_error: str | None = None


@dataclass
class LLMReply:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_message(self) -> dict:
        """The assistant message to append to the conversation (OpenAI format)."""
        # With tool calls and no text, content is null: the canonical form some providers insist on.
        message: dict[str, Any] = {"role": "assistant", "content": self.content or (None if self.tool_calls else "")}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.raw_arguments or json.dumps(call.arguments)},
                }
                for call in self.tool_calls
            ]
        return message


class LLMError(RuntimeError):
    """A failure the user can act on (bad key, unknown model, provider down...)."""


class ChatModel(Protocol):
    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply: ...


THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _clean_content(text: str | None) -> str | None:
    if text is None:
        return None
    return THINK_BLOCK.sub("", text).strip() or None


def parse_tool_call(call_id: str | None, name: str, raw: str | None) -> ToolCall:
    raw = raw or "{}"
    try:
        args = json.loads(raw)
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
        error = None
    except (json.JSONDecodeError, ValueError) as exc:
        args, error = {}, f"Could not parse the tool arguments as JSON ({exc}). Send valid JSON."
    return ToolCall(id=call_id or f"call_{uuid.uuid4().hex[:12]}", name=name, arguments=args, raw_arguments=raw,
                    parse_error=error)  # fmt: skip


# --- Real client -------------------------------------------------------------------


class OpenAICompatibleLLM:
    """Chat Completions with tool calling and exponential backoff on transient errors."""

    def __init__(
        self,
        settings: LLMSettings,
        client: Any = None,
        max_retries: int = 4,
        base_delay: float = 1.0,
        max_delay: float = 20.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        missing = settings.missing()
        if missing:
            raise LLMError("The AI analyst isn't set up yet: missing " + ", ".join(missing) + ".")
        self.settings = settings
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sleep = sleep
        if client is None:
            from openai import OpenAI

            try:
                client = OpenAI(
                    base_url=settings.base_url,
                    api_key=settings.api_key or "not-needed",
                    timeout=settings.timeout,
                    max_retries=0,  # retries are handled below, with backoff we control
                )
            except (OSError, ValueError) as exc:
                # e.g. an unreadable SSL_CERT_FILE or a malformed URL: report it, don't crash the page.
                raise LLMError(f"Couldn't set up the connection to {settings.provider_info.label}: {exc}") from exc
        self._client = client

    def _delay(self, attempt: int, exc: Exception) -> float:
        retry_after = None
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) or {}
        try:
            retry_after = float(headers.get("retry-after")) if headers.get("retry-after") else None
        except (TypeError, ValueError):
            retry_after = None
        backoff = min(self.max_delay, self.base_delay * (2**attempt)) + random.uniform(0, self.base_delay / 2)
        return min(self.max_delay, retry_after) if retry_after is not None else backoff

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
        import openai

        kwargs: dict[str, Any] = {"model": self.settings.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        transient = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                     openai.InternalServerError)  # fmt: skip
        attempt = 0
        while True:
            try:
                response = self._client.chat.completions.create(**kwargs)
                break
            except transient as exc:
                if attempt >= self.max_retries:
                    raise LLMError(_friendly(exc, self.settings)) from exc
                delay = self._delay(attempt, exc)
                log.warning("LLM call failed (%s); retry %d in %.1fs", type(exc).__name__, attempt + 1, delay)
                self._sleep(delay)
                attempt += 1
            except openai.APIError as exc:
                raise LLMError(_friendly(exc, self.settings)) from exc
        if not getattr(response, "choices", None):
            raise LLMError("The AI provider returned an empty response. Try again in a moment.")
        message = response.choices[0].message
        calls = [
            parse_tool_call(tc.id, tc.function.name, tc.function.arguments)
            for tc in (message.tool_calls or [])
            if getattr(tc, "function", None) is not None
        ]
        return LLMReply(content=_clean_content(message.content), tool_calls=calls)


def _friendly(exc: Exception, settings: LLMSettings) -> str:
    import openai

    provider = settings.provider_info.label
    if isinstance(exc, openai.AuthenticationError):
        return f"{provider} rejected the API key. Check LLM_API_KEY (or the key in ⚙️ AI settings)."
    if isinstance(exc, openai.PermissionDeniedError):
        return f"{provider} refused access to model '{settings.model}' for this API key."
    if isinstance(exc, openai.NotFoundError):
        return f"Model '{settings.model}' was not found at {settings.base_url}. Check LLM_MODEL."
    if isinstance(exc, openai.RateLimitError):
        return f"{provider} is rate-limiting requests (or the account is out of credit). Wait a minute and retry."
    if isinstance(exc, openai.APITimeoutError):
        return f"{provider} took too long to respond. Try again."
    if isinstance(exc, openai.APIConnectionError):
        return f"Couldn't reach {settings.base_url}. Check the URL and your internet connection."
    if isinstance(exc, openai.BadRequestError):
        text = str(exc)
        if "tool" in text.lower() or "function" in text.lower():
            return f"Model '{settings.model}' doesn't support tool calling. Choose a model that does."
        return f"{provider} rejected the request: {text[:300]}"
    return f"{provider} error: {str(exc)[:300]}"


def build_client(settings: LLMSettings) -> ChatModel:
    """Factory used by the app (tests replace it with a MockLLM)."""
    return OpenAICompatibleLLM(settings)


def check_connection(model: ChatModel) -> str:
    reply = model.complete([{"role": "user", "content": "Reply with the single word: ready"}])
    return (reply.content or "").strip()[:60] or "(empty reply)"


# --- Mock --------------------------------------------------------------------------

Script = LLMReply | Callable[[list[dict], list[dict] | None], LLMReply]


class MockLLM:
    """Scripted stand-in for tests and offline development. Records every payload."""

    def __init__(self, script: list[Script], repeat_last: bool = False) -> None:
        self.script = list(script)
        self.repeat_last = repeat_last
        self.calls: list[dict] = []
        self.settings = replace(LLMSettings(), provider="custom", model="mock", base_url="mock://")

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
        self.calls.append({"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools)})
        if not self.script:
            raise LLMError("MockLLM script exhausted.")
        step = self.script[0] if (self.repeat_last and len(self.script) == 1) else self.script.pop(0)
        reply = step(messages, tools) if callable(step) else copy.deepcopy(step)
        for call in reply.tool_calls:
            call.raw_arguments = call.raw_arguments or json.dumps(call.arguments)
        return reply

    def payload_text(self) -> str:
        """Everything that was sent to the 'model', as one string (for privacy assertions)."""
        return json.dumps(self.calls, default=str)


def tool_reply(name: str, content: str | None = None, **arguments: Any) -> LLMReply:
    return LLMReply(content=content, tool_calls=[ToolCall(f"call_{uuid.uuid4().hex[:8]}", name, arguments)])


def text_reply(content: str) -> LLMReply:
    return LLMReply(content=content)
