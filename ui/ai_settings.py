"""⚙️ AI settings: which LLM the analyst uses.

Defaults come from environment variables / Streamlit secrets (for deployment).
Anything typed in the sidebar applies to this browser session only and is
never written to disk or logs.
"""

from __future__ import annotations

import streamlit as st

from agent import llm as llm_mod
from agent.llm import PROVIDERS, LLMError, LLMSettings

KEYS = {"provider": "ai_provider", "api_key": "ai_api_key", "model": "ai_model", "base_url": "ai_base_url"}


def _secrets():
    try:
        return st.secrets
    except Exception:  # noqa: BLE001 - no secrets file
        return None


def env_settings() -> LLMSettings:
    return llm_mod.settings_from_env(_secrets())


def init() -> None:
    if KEYS["provider"] in st.session_state:
        return
    env = env_settings()
    st.session_state[KEYS["provider"]] = env.provider
    st.session_state[KEYS["model"]] = env.model
    st.session_state[KEYS["base_url"]] = env.base_url
    st.session_state[KEYS["api_key"]] = ""
    st.session_state["ai_test_message"] = None


def _on_provider_change() -> None:
    provider = st.session_state[KEYS["provider"]]
    env = env_settings()
    source = env if provider == env.provider else llm_mod.settings_for_provider(provider)
    st.session_state[KEYS["model"]] = source.model
    st.session_state[KEYS["base_url"]] = source.base_url
    st.session_state["ai_test_message"] = None


def current() -> LLMSettings:
    """Effective settings: sidebar values, falling back to the environment's key."""
    init()
    env = env_settings()
    provider = st.session_state[KEYS["provider"]]
    typed_key = str(st.session_state.get(KEYS["api_key"]) or "").strip()
    key = typed_key or (env.api_key if provider == env.provider else "")
    return llm_mod.settings_for_provider(
        provider,
        api_key=key,
        model=str(st.session_state.get(KEYS["model"]) or ""),
        base_url=str(st.session_state.get(KEYS["base_url"]) or ""),
    )


def key_source() -> str:
    if str(st.session_state.get(KEYS["api_key"]) or "").strip():
        return "key entered in the sidebar (this session only)"
    if current().api_key:
        return "key from the server's environment / secrets"
    return "no key needed" if not current().needs_key else "no key yet"


def _test() -> None:
    settings = current()
    try:
        reply = llm_mod.check_connection(llm_mod.build_client(settings))
        st.session_state["ai_test_message"] = ("success", f"Connected to {settings.describe()} — replied “{reply}”.")
    except LLMError as exc:
        st.session_state["ai_test_message"] = ("error", str(exc))


def render(expanded: bool = False) -> None:
    init()
    settings = current()
    with st.sidebar:
        label = "⚙️ AI settings" + ("" if settings.configured else " · not set up")
        with st.expander(label, expanded=expanded):
            st.selectbox(
                "AI provider",
                list(PROVIDERS),
                format_func=lambda k: PROVIDERS[k].label,
                key=KEYS["provider"],
                on_change=_on_provider_change,
            )
            provider = PROVIDERS[st.session_state[KEYS["provider"]]]
            if provider.needs_key:
                env_has_key = bool(env_settings().api_key) and env_settings().provider == provider.key
                st.text_input(
                    "API key",
                    type="password",
                    key=KEYS["api_key"],
                    placeholder="Using the server's key" if env_has_key else "Paste your API key",
                    help="Used for this browser session only; never saved.",
                )
                if provider.key_url:
                    st.caption(f"Get a key: [{provider.key_url}]({provider.key_url})")
            else:
                st.caption(f"Runs on your own computer. Install Ollama, then `ollama pull {provider.default_model}`.")
            st.text_input("Model", key=KEYS["model"], help="Any model on this provider that supports tool calling.")
            if provider.key in ("custom", "ollama"):
                st.text_input("Base URL", key=KEYS["base_url"], placeholder="https://…/v1")
            settings = current()
            if settings.configured:
                st.caption(f"✅ Ready: {settings.describe()} · {key_source()}")
                st.button("Test connection", key="ai_test", on_click=_test, width="stretch")
            else:
                st.caption("Missing: " + ", ".join(m.replace("LLM_", "").replace("_", " ").lower()
                                                   for m in settings.missing()))  # fmt: skip
            message = st.session_state.get("ai_test_message")
            if message:
                (st.success if message[0] == "success" else st.error)(message[1])
