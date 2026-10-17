"""Validate and execute AI-written analysis code safely.

Layers, outermost first:

1. ``validate_python`` / ``validate_sql`` reject anything outside a small,
   read-only analysis subset *before* it runs (AST and DuckDB parser checks).
2. Code runs in a separate Python process (``agent/_worker.py``) started in
   isolated mode with an empty environment, so it cannot see API keys.
3. That process has restricted builtins, CPU / memory / file-size limits and
   a wall-clock alarm; DuckDB runs with external access disabled.
4. Results come back as JSON text, never pickles, and the parent kills the
   process if it overruns.
"""

from __future__ import annotations

import ast
import json
import os
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

WORKER = Path(__file__).with_name("_worker.py")
MARKER = "__GRADELENS_RESULT__"
DEFAULT_TIMEOUT = 10
STARTUP_BUDGET = 25  # seconds for interpreter start + imports, on top of the code's own timeout

PREINJECTED = {"df", "pd", "np", "px", "go", "math"}
BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "input", "breakpoint", "help", "exit", "quit", "memoryview", "type", "super",
    "classmethod", "staticmethod", "property",
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib", "builtins", "ctypes", "pickle",
    "marshal", "io", "codecs", "signal", "resource", "threading", "multiprocessing", "requests", "urllib",
}  # fmt: skip
BANNED_ATTRS = {
    # file / network I/O
    "to_csv", "to_excel", "to_pickle", "to_parquet", "to_feather", "to_hdf", "to_sql", "to_stata", "to_orc",
    "to_clipboard", "to_html", "to_json", "to_xml", "to_latex", "to_markdown", "to_gbq",
    "read_csv", "read_excel", "read_json", "read_html", "read_xml", "read_parquet", "read_feather",
    "read_pickle", "read_sql", "read_sql_query", "read_sql_table", "read_table", "read_fwf", "read_clipboard",
    "read_hdf", "read_sas", "read_spss", "read_stata", "read_orc", "read_gbq",
    "load", "loads", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt", "fromfile",
    "tofile", "memmap", "fromregex", "DataSource", "write_html", "write_image", "write_json", "show",
    # code execution / introspection escapes
    "eval", "query", "exec", "system", "popen", "spawn", "format", "format_map", "mro",
    "ctypes", "ctypeslib", "lib", "io", "os", "sys", "subprocess", "builtins", "importlib", "shutil",
    "socket", "pathlib", "pickle", "marshal", "codecs", "f2py", "testing", "distutils", "core", "compat",
    "plotting", "plot", "pipe_to", "ExcelWriter", "ExcelFile", "HDFStore", "webbrowser", "to_image",
    "full_figure_for_development",
    "from_file", "read", "write",
}  # fmt: skip
WRITE_KWARGS = {"buf", "path", "path_or_buf", "excel_writer", "fname", "file", "filepath_or_buffer"}
BANNED_NODES = (
    ast.Import, ast.ImportFrom, ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith,
    ast.Global, ast.Nonlocal,
)  # fmt: skip
SQL_DENYLIST = (
    "read_csv", "read_parquet", "read_json", "read_text", "read_blob", "read_xlsx", "read_ndjson", "glob(",
    "parquet_scan", "csv_scan", "sniff_csv", "copy ", "attach", "install", "load ", "pragma", "export ",
    "http://", "https://", "s3://", "gcs://", "file:", "getenv", "current_setting",
)  # fmt: skip


class SandboxViolation(ValueError):
    """The code was rejected before running."""


@dataclass
class ExecutionResult:
    ok: bool
    table: pd.DataFrame | None = None
    total_rows: int = 0
    scalar: object = None
    kind: str = "none"
    figure_json: str | None = None
    stdout: str = ""
    error: str | None = None
    notes: list[str] = field(default_factory=list)


class _Validator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.problems: list[str] = []

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, BANNED_NODES):
            kind = "import" if isinstance(node, (ast.Import, ast.ImportFrom)) else type(node).__name__
            self.problems.append(f"line {getattr(node, 'lineno', '?')}: '{kind}' is not allowed")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in BANNED_NAMES or node.id.startswith("__"):
            self.problems.append(f"line {node.lineno}: use of '{node.id}' is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            self.problems.append(f"line {node.lineno}: private/dunder attribute '.{node.attr}' is not allowed")
        elif node.attr in BANNED_ATTRS:
            self.problems.append(f"line {node.lineno}: '.{node.attr}' is not allowed (no file, network or eval)")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        for kw in node.keywords:
            if kw.arg in WRITE_KWARGS:
                self.problems.append(f"line {node.lineno}: writing to '{kw.arg}' is not allowed")
        if isinstance(node.func, ast.Attribute) and node.func.attr == "to_string" and node.args:
            self.problems.append(f"line {node.lineno}: to_string() may not take a file argument")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name.startswith("__"):
            self.problems.append(f"line {node.lineno}: dunder function names are not allowed")
        if node.decorator_list:
            self.problems.append(f"line {node.lineno}: decorators are not allowed")
        self.generic_visit(node)


def _ensure_result_assignment(tree: ast.Module) -> ast.Module:
    """If the code never assigns `result` but ends in an expression, use that expression."""
    assigns_result = any(
        isinstance(node, ast.Name) and node.id == "result" and isinstance(node.ctx, ast.Store)
        for node in ast.walk(tree)
    )
    if not assigns_result and tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        tree.body[-1] = ast.copy_location(
            ast.Assign(targets=[ast.Name(id="result", ctx=ast.Store())], value=last.value), last
        )
        ast.fix_missing_locations(tree)
    return tree


def validate_python(code: str) -> str:
    """Return (possibly normalised) code, or raise SandboxViolation."""
    if not code or not code.strip():
        raise SandboxViolation("No code was provided.")
    if len(code) > 20_000:
        raise SandboxViolation("Code is too long (limit 20,000 characters).")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxViolation(f"SyntaxError: {exc.msg} (line {exc.lineno})") from exc
    validator = _Validator()
    validator.visit(tree)
    if validator.problems:
        unique = list(dict.fromkeys(validator.problems))
        raise SandboxViolation(
            "Code rejected by the sandbox: " + "; ".join(unique[:5])
            + ". Use only pandas/numpy/plotly on `df` (pd, np, px, go are already imported)."
        )  # fmt: skip
    return ast.unparse(_ensure_result_assignment(tree))


def validate_sql(query: str) -> str:
    if not query or not query.strip():
        raise SandboxViolation("No SQL was provided.")
    try:
        statements = duckdb.extract_statements(query)
    except duckdb.Error as exc:
        raise SandboxViolation(f"SQL syntax error: {exc}") from exc
    if len(statements) != 1:
        raise SandboxViolation("Send exactly one SQL statement.")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise SandboxViolation("Only read-only SELECT / WITH queries are allowed.")
    lowered = " ".join(query.lower().split())
    for word in SQL_DENYLIST:
        if word in lowered:
            raise SandboxViolation(f"SQL uses '{word.strip()}', which is not allowed. Query the table `marks` only.")
    return query.strip().rstrip(";")


def _worker_env() -> dict[str, str]:
    env = {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    if os.name == "nt":  # Windows needs these to start Python at all
        for key in ("SYSTEMROOT", "TEMP", "TMP"):
            if key in os.environ:
                env[key] = os.environ[key]
    return env


def _parse_output(stdout: bytes) -> dict | None:
    text = stdout.decode("utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER) :])
    return None


def _to_result(payload: dict) -> ExecutionResult:
    if not payload.get("ok"):
        return ExecutionResult(ok=False, error=payload.get("error", "Unknown error"))
    result = payload.get("result") or {"kind": "none"}
    out = ExecutionResult(
        ok=True, kind=result["kind"], figure_json=payload.get("figure"), stdout=payload.get("stdout", "")
    )
    if result["kind"] == "dataframe":
        out.table = pd.DataFrame(result["data"], columns=result["columns"])
        out.total_rows = int(result.get("rows", len(out.table)))
    elif result["kind"] in ("scalar", "json", "text"):
        out.scalar = result.get("value")
    if payload.get("figure_ignored"):
        out.notes.append("`fig` was not a Plotly figure and was ignored.")
    return out


def execute(
    mode: str,
    code: str,
    df: pd.DataFrame,
    timeout: int = DEFAULT_TIMEOUT,
    figure_style: dict | None = None,
) -> ExecutionResult:
    """Validate and run code in the sandbox process. Never raises for bad user code.

    ``figure_style`` (``colorway`` list, ``font_family``) makes AI-built Plotly charts match the app.
    """
    try:
        checked = validate_sql(code) if mode == "sql" else validate_python(code)
    except SandboxViolation as exc:
        return ExecutionResult(ok=False, error=str(exc))
    job = {
        "mode": mode,
        "code": checked,
        "df": df.copy(),
        "timeout": int(timeout),
        "needs_plotly": "px." in checked or "go." in checked,
        "figure_style": dict(figure_style or {}),
    }
    with tempfile.TemporaryDirectory(prefix="gradelens-sandbox-") as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-B", str(WORKER)],
                input=pickle.dumps(job, protocol=pickle.HIGHEST_PROTOCOL),
                capture_output=True,
                timeout=timeout + STARTUP_BUDGET,
                env=_worker_env(),
                cwd=workdir,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(ok=False, error=f"Code took longer than {timeout} seconds and was stopped.")
    payload = _parse_output(proc.stdout)
    if payload is None:
        reason = _describe_exit(proc.returncode, timeout)
        return ExecutionResult(ok=False, error=reason)
    return _to_result(payload)


def _describe_exit(returncode: int, timeout: int) -> str:
    signals = {-9: "it used too much memory or time", -24: f"it ran longer than {timeout} seconds of CPU time",
               -25: "it tried to write a file", -11: "it crashed"}  # fmt: skip
    why = signals.get(returncode, f"exit code {returncode}")
    return f"The code was stopped by the sandbox ({why})."
