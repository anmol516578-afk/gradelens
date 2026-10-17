"""Sandboxed executor for AI-written analysis code. Runs as a separate process.

Started by ``agent.sandbox`` as ``python -I -B _worker.py`` with an empty
environment (no API keys) and the job pickled on stdin. Untrusted code never
runs in the app process, and results come back as JSON only — the parent
never unpickles anything this process produces.

Defence in depth (the parent has already validated the code with ``ast``):
restricted builtins, CPU-time and file-size limits, an alarm for the
wall-clock timeout, and a DuckDB connection with external access disabled.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import pickle
import sys
import traceback

MARKER = "__GRADELENS_RESULT__"
MAX_ROWS = 5000
MAX_STDOUT = 4000

SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bin", "bool", "bytes", "callable", "chr", "complex", "dict", "divmod", "enumerate",
    "filter", "float", "format", "frozenset", "hash", "hex", "int", "isinstance", "issubclass", "iter", "len",
    "list", "map", "max", "min", "next", "oct", "ord", "pow", "print", "range", "repr", "reversed", "round",
    "set", "slice", "sorted", "str", "sum", "tuple", "zip", "object", "hasattr", "True", "False", "None",
    "Exception", "ArithmeticError", "AttributeError", "IndexError", "KeyError", "LookupError", "NameError",
    "RuntimeError", "StopIteration", "TypeError", "ValueError", "ZeroDivisionError",
)  # fmt: skip


def _apply_limits(cpu_seconds: int) -> None:
    try:
        import resource
    except ImportError:  # Windows: rely on the parent's timeout
        return
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))  # writing any regular file kills the process
    with contextlib.suppress(OSError, ValueError):
        # Cap memory at what the libraries already use plus 1.5 GiB, so a runaway allocation
        # fails with MemoryError instead of starving the server.
        with open("/proc/self/status") as status:
            vm_kb = next(int(line.split()[1]) for line in status if line.startswith("VmSize:"))
        limit = vm_kb * 1024 + 1536 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _start_alarm(seconds: int) -> None:
    try:
        import signal
    except ImportError:
        return
    if not hasattr(signal, "SIGALRM"):
        return

    def _timeout(signum, frame):
        raise TimeoutError(f"Code took longer than {seconds} seconds (possible infinite loop).")

    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(seconds)


def _stop_alarm() -> None:
    with contextlib.suppress(ImportError, AttributeError):
        import signal

        signal.alarm(0)


def _json_safe(value):
    import numpy as np

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _frame_payload(frame) -> dict:
    import pandas as pd

    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = [" ".join(str(p) for p in col if str(p)) for col in frame.columns]
    # Keep a meaningful index (named, multi-level or labels such as subjects) as columns, but drop
    # plain row numbers left over from sorting or filtering (see DEBUG_LOG #10).
    named = any(name is not None for name in frame.index.names)
    if isinstance(frame.index, pd.MultiIndex) or named or not pd.api.types.is_integer_dtype(frame.index):
        frame = frame.reset_index()
    else:
        frame = frame.reset_index(drop=True)
    frame.columns = [str(c) for c in frame.columns]
    total = len(frame)
    frame = frame.head(MAX_ROWS)
    data = json.loads(frame.to_json(orient="split", index=False, date_format="iso", default_handler=str))
    return {"kind": "dataframe", "columns": data["columns"], "data": data["data"], "rows": total}


def _result_payload(result) -> dict:
    import pandas as pd

    if result is None:
        return {"kind": "none"}
    if isinstance(result, pd.DataFrame):
        return _frame_payload(result)
    if isinstance(result, pd.Series):
        name = str(result.name) if result.name is not None else "value"
        return _frame_payload(result.to_frame(name=name))
    if isinstance(result, pd.Index):
        return _frame_payload(pd.DataFrame({"value": list(result)}))
    value = _json_safe(result)
    if isinstance(value, (int, float, str, bool)):
        return {"kind": "scalar", "value": value}
    try:
        return {"kind": "json", "value": json.loads(json.dumps(result, default=str))}
    except (TypeError, ValueError):
        return {"kind": "text", "value": repr(result)[:2000]}


def _user_error(exc: BaseException) -> str:
    lines = traceback.extract_tb(exc.__traceback__)
    where = [f"line {fr.lineno}" for fr in lines if fr.filename == "<analysis>"]
    location = f" ({where[-1]})" if where else ""
    return f"{type(exc).__name__}: {exc}{location}"


def _apply_figure_style(style: dict) -> None:
    """Default template so charts the AI builds use the app's palette and font."""
    if not style:
        return
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio

    layout = {"colorway": style.get("colorway") or None, "font": {"family": style.get("font_family") or None}}
    pio.templates["gradelens"] = go.layout.Template(layout=layout)
    pio.templates.default = "plotly_white+gradelens"
    if style.get("colorway"):
        px.defaults.color_discrete_sequence = list(style["colorway"])


def run_python(job: dict) -> dict:
    import numpy as np
    import pandas as pd

    namespace = {"__builtins__": {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
                                  for n in SAFE_BUILTIN_NAMES}}  # fmt: skip
    namespace.update({"df": job["df"], "pd": pd, "np": np, "math": math})
    if job.get("needs_plotly"):
        import plotly.express as px
        import plotly.graph_objects as go

        _apply_figure_style(job.get("figure_style") or {})
        namespace.update({"px": px, "go": go})
    code = compile(job["code"], "<analysis>", "exec")
    printed = io.StringIO()
    _apply_limits(job["timeout"] + 2)
    _start_alarm(job["timeout"])
    try:
        with contextlib.redirect_stdout(printed):
            exec(code, namespace)  # noqa: S102 - validated, restricted, isolated process
    finally:
        _stop_alarm()
    figure = namespace.get("fig")
    figure_json = None
    if figure is not None:
        to_json = getattr(figure, "to_json", None)
        figure_json = to_json() if callable(to_json) and type(figure).__name__ == "Figure" else None
    return {
        "ok": True,
        "result": _result_payload(namespace.get("result")),
        "figure": figure_json,
        "figure_ignored": figure is not None and figure_json is None,
        "stdout": printed.getvalue()[:MAX_STDOUT],
    }


def run_sql(job: dict) -> dict:
    import duckdb

    con = duckdb.connect(
        config={
            "enable_external_access": False,
            "autoinstall_known_extensions": False,
            "autoload_known_extensions": False,
            "lock_configuration": True,
        }
    )
    con.register("marks", job["df"])
    _apply_limits(job["timeout"] + 2)
    _start_alarm(job["timeout"])
    try:
        frame = con.execute(job["code"]).df()
    finally:
        _stop_alarm()
    return {"ok": True, "result": _frame_payload(frame), "figure": None, "figure_ignored": False, "stdout": ""}


def main() -> int:
    job = pickle.load(sys.stdin.buffer)  # trusted: written by the parent process
    try:
        output = run_sql(job) if job["mode"] == "sql" else run_python(job)
    except BaseException as exc:  # report every failure of untrusted code back to the agent
        output = {"ok": False, "error": _user_error(exc)}
    sys.stdout.write("\n" + MARKER + json.dumps(output, default=str) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
