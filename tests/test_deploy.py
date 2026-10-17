"""Deployment configuration: the container start command, health check and pins.

The image itself is built and run in QA (see DEBUG_LOG); these tests stop the config from regressing.
"""

from __future__ import annotations

import json
import re
import subprocess

from tests.conftest import ROOT

DOCKERFILE = (ROOT / "Dockerfile").read_text()


def _instruction(name: str) -> str:
    """The full text of a Dockerfile instruction, joining backslash-continued lines."""
    joined = re.sub(r"\\\n\s*", " ", DOCKERFILE)
    matches = [line[len(name) + 1 :].strip() for line in joined.splitlines() if line.startswith(name + " ")]
    assert matches, f"no {name} instruction"
    return matches[-1]


def _start_command(env: dict[str, str]) -> list[str]:
    """Run the CMD's shell snippet with `exec streamlit` swapped for printing its arguments."""
    argv = json.loads(_instruction("CMD"))
    assert argv[:2] == ["sh", "-c"]
    script = argv[2].replace("exec streamlit", "printf '%s\\n'", 1)
    out = subprocess.run(["sh", "-c", script], env=env, capture_output=True, text=True, check=True).stdout
    return out.split()


def test_start_command_honours_port():
    assert "--server.port=8501" in _start_command({})
    assert "--server.port=10000" in _start_command({"PORT": "10000"})
    assert "--server.address=0.0.0.0" in _start_command({})


def test_xsrf_is_relaxed_only_on_hugging_face_spaces():
    """Spaces embeds the app in an iframe where the XSRF cookie is blocked and uploads get 403."""
    assert not any("enableXsrfProtection" in a for a in _start_command({}))
    assert "--server.enableXsrfProtection=false" in _start_command({"SPACE_ID": "user/gradelens"})
    assert not any("enableXsrfProtection" in a for a in _start_command({"SPACE_ID": ""}))


def test_container_runs_unprivileged_with_an_explicit_group():
    assert _instruction("USER") == "appuser"
    run_lines = " ".join(re.sub(r"\\\n\s*", " ", DOCKERFILE).splitlines())
    assert "groupadd --gid 1000 appuser" in run_lines
    assert "useradd --create-home --uid 1000 --gid appuser appuser" in run_lines


def test_production_settings_in_image():
    """No file watcher in production: without watchdog Streamlit would poll the source files forever."""
    joined = re.sub(r"\\\n\s*", " ", DOCKERFILE)
    assert "STREAMLIT_SERVER_FILE_WATCHER_TYPE=none" in joined
    assert "STREAMLIT_SERVER_HEADLESS=true" in joined


def test_healthcheck_command_is_valid_python():
    health = _instruction("HEALTHCHECK")
    code = re.search(r'python -c "(.*)"$', health).group(1)
    compile(code, "<healthcheck>", "exec")
    assert "_stcore/health" in code and "PORT" in code


def test_runtime_requirements_are_exact_pins():
    lines = [line.split("#")[0].strip() for line in (ROOT / "requirements.txt").read_text().splitlines()]
    pins = [line for line in lines if line]
    assert pins and all(re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][0-9A-Za-z.]*", p) for p in pins), pins
