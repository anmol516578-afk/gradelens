"""Real OpenAI SDK + real agent loop + real sandbox, against a local fake server over HTTP."""

from __future__ import annotations

import pytest

from agent.analyst import ANSWERED, AnalystAgent
from agent.llm import LLMError, LLMSettings, OpenAICompatibleLLM, check_connection
from agent.masking import Pseudonymizer
from tests import fake_openai_server


@pytest.fixture(scope="module")
def server():
    srv, url = fake_openai_server.start()
    yield url
    srv.shutdown()


def settings(url: str, key: str = "sk-local") -> LLMSettings:
    return LLMSettings(provider="custom", base_url=url, api_key=key, model="fake")


def test_full_round_trip_with_error_and_fix(server, derived, config):
    pseudo = Pseudonymizer.from_frame(derived)
    fake_openai_server.Handler.calls.clear()
    agent = AnalystAgent(OpenAICompatibleLLM(settings(server)), pseudo.mask_frame(derived), config)
    result = agent.run("Which section has the highest fail rate?")
    assert result.status == ANSWERED, result.answer
    assert [s.ok for s in result.steps] == [False, True]
    assert result.answer.startswith("**Section B** has the highest fail rate")
    assert result.figure_json
    # Real HTTP payloads carried tool schemas and no student names.
    sent = str(fake_openai_server.Handler.calls)
    assert '"run_python"' in sent or "'run_python'" in sent
    assert not any(name in sent for name in pseudo.real_names())


def test_connection_check_and_bad_key(server):
    assert check_connection(OpenAICompatibleLLM(settings(server))) == "ready"
    with pytest.raises(LLMError, match="rejected the API key"):
        OpenAICompatibleLLM(settings(server, key="bad-key")).complete([{"role": "user", "content": "hi"}])


def test_unreachable_server_is_reported():
    llm = OpenAICompatibleLLM(settings("http://127.0.0.1:9/v1"), max_retries=1, base_delay=0.01)
    with pytest.raises(LLMError, match="Couldn't reach"):
        llm.complete([{"role": "user", "content": "hi"}])
