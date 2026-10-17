"""A tiny OpenAI-compatible Chat Completions server for integration tests and demos.

It plays a deterministic "model": first it writes buggy pandas code, then it
fixes the bug after seeing the error, then it calls ``final_answer`` using
the numbers it got back. It speaks the real wire format, so the real OpenAI
SDK, the real agent loop and the real sandbox are all exercised.

Run standalone (for a browser demo without an API key):

    python tests/fake_openai_server.py 8765
    LLM_PROVIDER=custom LLM_BASE_URL=http://127.0.0.1:8765/v1 LLM_MODEL=fake streamlit run app.py
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BUGGY = "graded = df[df['result'].isin(['Pass', 'Fail'])]\nresult = graded.groupby('sec')['result'].count()"
FIXED = (
    "graded = df[df['result'].isin(['Pass', 'Fail'])]\n"
    "result = (graded.assign(failed=graded['result'].eq('Fail'))\n"
    "          .groupby('section', as_index=False)['failed'].mean())\n"
    "result['fail_rate'] = (result.pop('failed') * 100).round(1)\n"
    "result = result.sort_values('fail_rate', ascending=False)\n"
    "fig = px.bar(result, x='section', y='fail_rate', title='Fail rate by section (%)',\n"
    "             labels={'fail_rate': 'Fail rate (%)', 'section': 'Section'})"
)
WEAKEST = (
    "per_student = df.groupby('student_id', as_index=False)['score_pct'].mean().round(1)\n"
    "result = per_student.nsmallest(5, 'score_pct')\n"
    "fig = px.bar(result, x='student_id', y='score_pct', title='Five weakest students (average %)')"
)


def _tool_call(name: str, arguments: dict) -> dict:
    return {
        "id": f"call_{uuid.uuid4().hex[:10]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def decide(messages: list[dict]) -> dict:
    """Return the assistant message for this point in the conversation."""
    question = next(
        m["content"] for m in reversed(messages) if m["role"] == "user" and "final_answer now" not in m["content"]
    )
    last = messages[-1]
    weakest = "weak" in question.lower() or "support" in question.lower()
    if last["role"] == "user":
        if weakest:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("run_python", {"code": WEAKEST, "purpose": "Lowest average scores"})],
            }
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("run_python", {"code": BUGGY, "purpose": "Fail rate by section"})],
        }
    content = last.get("content") or ""
    if last["role"] == "tool" and content.startswith("Error"):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("run_python", {"code": FIXED, "purpose": "Fail rate by section (fixed)"})],
        }
    rows = re.findall(r"^\s*(\S+)\s+([\d.]+)\s*$", content, flags=re.M)
    if weakest and rows:
        ids = ", ".join(r[0] for r in rows[:3])
        answer = (
            f"The students needing the most support are **{ids}** (lowest averages: "
            + ", ".join(f"{r[1]}%" for r in rows[:3])
            + ")."
        )
        assumptions = ["Average of score_pct across all subjects in view"]
    elif rows:
        top = rows[0]
        answer = (
            f"**Section {top[0]}** has the highest fail rate at **{top[1]}%**. "
            + "; ".join(f"Section {s}: {r}%" for s, r in rows[1:])
            + "."
        )
        assumptions = ["Fail rate counts only graded (Pass/Fail) rows"]
    else:
        answer, assumptions = "I could not compute that from the data.", []
    return {"role": "assistant", "content": None,
            "tool_calls": [_tool_call("final_answer", {"answer": answer, "assumptions": assumptions})]}  # fmt: skip


def summarize(prompt: str) -> str:
    """Deterministic stand-in for an AI-written executive summary: restates the facts it was given."""
    facts = [line[2:] for line in prompt.splitlines() if line.startswith("- ")]
    body = " ".join(facts[1:4])
    return f"{body} Recommended next steps: run remedial sessions for the weakest subject and meet at-risk students."


class Handler(BaseHTTPRequestHandler):
    calls: list[dict] = []

    def log_message(self, *args) -> None:  # keep test output quiet
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        Handler.calls.append(body)
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)
            return
        if self.headers.get("authorization") == "Bearer bad-key":
            self._send(401, {"error": {"message": "Incorrect API key provided", "type": "invalid_request_error"}})
            return
        messages = body.get("messages", [])
        if not body.get("tools"):
            prompt = messages[-1].get("content", "") if messages else ""
            message = {"role": "assistant", "content": summarize(prompt) if "executive summary" in prompt else "ready"}
        else:
            message = decide(messages)
        self._send(
            200,
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "fake"),
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def start(port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


if __name__ == "__main__":
    srv, url = start(int(sys.argv[1]) if len(sys.argv) > 1 else 8765)
    print(f"Fake OpenAI-compatible server at {url}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
