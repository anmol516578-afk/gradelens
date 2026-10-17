"""The sandbox must block every escape route and still run ordinary analysis code."""

from __future__ import annotations

import json
import time

import pandas as pd
import pytest

from agent import sandbox
from agent.sandbox import SandboxViolation, execute, validate_python, validate_sql


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_id": ["S001", "S002", "S003", "S004"],
            "section": ["A", "A", "B", "B"],
            "score_pct": [80.0, 35.0, 55.0, 20.0],
            "result": ["Pass", "Fail", "Pass", "Fail"],
        }
    )


# --- Required blocked cases (spec section 9) ---------------------------------------

BLOCKED_PYTHON = {
    "open_file": "open('x')",
    "open_write": "open('x', 'w').write('pwned')",
    "dunder_import": "__import__('os')",
    "os_system": "os.system('echo hi')",
    "import_os": "import os\nos.system('echo hi')",
    "from_import": "from subprocess import run",
    "to_csv": "df.to_csv('f')",
    "to_excel": "df.to_excel('f.xlsx')",
    "getattr_trick": "getattr(df, 'to_csv')('f')",
    "getattr_builtins": "getattr(__builtins__, 'open')('x')",
    "dunder_class_escape": "().__class__.__bases__[0].__subclasses__()",
    "dunder_globals": "f = lambda: 0\nf.__globals__",
    "format_escape": "'{0.__class__}'.format(df)",
    "eval": "eval('1+1')",
    "exec": "exec('x = 1')",
    "compile": "compile('1', 'a', 'eval')",
    "df_eval": "df.eval('score_pct * 2')",
    "pd_eval": "pd.eval('1+1')",
    "df_query": "df.query('score_pct > 50')",
    "read_csv": "pd.read_csv('/etc/passwd')",
    "np_load": "np.load('x.npy')",
    "print_to_file": "print('x', file=None)",
    "type_builtin": "type(df)",
    "vars_builtin": "vars()",
    "globals_builtin": "globals()",
    "private_attr": "df._data",
    "pandas_io": "pd.io.common",
    "numpy_ctypes": "np.ctypeslib",
    "excel_writer": "pd.ExcelWriter('x.xlsx')",
    "fig_write": "fig = px.bar(df, x='section', y='score_pct')\nfig.write_html('x.html')",
    "class_def": "class X:\n    pass",
    "global_stmt": "def f():\n    global df\n    return df",
}


@pytest.mark.parametrize("code", BLOCKED_PYTHON.values(), ids=BLOCKED_PYTHON.keys())
def test_dangerous_python_is_rejected_before_running(code):
    with pytest.raises(SandboxViolation):
        validate_python(code)


@pytest.mark.parametrize("code", BLOCKED_PYTHON.values(), ids=BLOCKED_PYTHON.keys())
def test_execute_reports_rejection_without_raising(code, df):
    result = execute("python", code, df)
    assert not result.ok
    assert "not allowed" in result.error or "rejected" in result.error


def test_infinite_loop_is_killed_by_timeout(df):
    start = time.monotonic()
    result = execute("python", "while True:\n    pass", df, timeout=2)
    elapsed = time.monotonic() - start
    assert not result.ok
    assert "longer than 2 seconds" in result.error
    assert elapsed < 15


def test_loop_that_swallows_the_timeout_is_still_killed(df):
    code = "while True:\n    try:\n        while True:\n            pass\n    except:\n        pass"
    start = time.monotonic()
    result = execute("python", code, df, timeout=1)
    assert not result.ok
    assert "stopped" in result.error
    assert time.monotonic() - start < 20  # CPU limit ends it, long before the parent's hard timeout


BLOCKED_SQL = {
    "delete": "DELETE FROM marks",
    "drop": "DROP TABLE marks",
    "insert": "INSERT INTO marks VALUES (1)",
    "update": "UPDATE marks SET score_pct = 100",
    "create": "CREATE TABLE t AS SELECT * FROM marks",
    "copy": "COPY marks TO 'out.csv'",
    "attach": "ATTACH 'other.db'",
    "pragma": "PRAGMA database_list",
    "install": "INSTALL httpfs",
    "two_statements": "SELECT 1; DROP TABLE marks",
    "read_file": "SELECT * FROM read_csv('/etc/passwd')",
    "read_text": "SELECT * FROM read_text('/etc/passwd')",
    "url": "SELECT * FROM 'https://example.com/data.csv'",
    "getenv": "SELECT getenv('LLM_API_KEY')",
}


@pytest.mark.parametrize("query", BLOCKED_SQL.values(), ids=BLOCKED_SQL.keys())
def test_non_select_sql_is_rejected(query, df):
    with pytest.raises(SandboxViolation):
        validate_sql(query)
    assert not execute("sql", query, df).ok


def test_file_access_blocked_inside_duckdb_even_if_validation_missed(df, monkeypatch):
    """Defence in depth: DuckDB runs with external access disabled."""
    monkeypatch.setattr(sandbox, "validate_sql", lambda q: q)
    result = execute("sql", "SELECT * FROM read_text('/etc/hostname')", df)
    assert not result.ok
    assert "disabled" in result.error.lower() or "permission" in result.error.lower()


def test_restricted_builtins_even_if_validation_missed(df, monkeypatch):
    """Defence in depth: the worker process has no open() or __import__."""
    monkeypatch.setattr(sandbox, "validate_python", lambda c: c)
    for code in ("result = open('x', 'w')", "import os\nresult = os.getcwd()"):
        result = execute("python", code, df)
        assert not result.ok, code
        assert "NameError" in result.error or "ImportError" in result.error or "import" in result.error


def test_worker_cannot_see_api_keys(df, monkeypatch):
    """The worker starts with an empty environment, so secrets never reach untrusted code."""
    monkeypatch.setenv("LLM_API_KEY", "sk-secret-value")
    assert "LLM_API_KEY" not in sandbox._worker_env()


# --- Ordinary analysis still works ----------------------------------------------


def test_series_result_becomes_table(df):
    result = execute("python", "result = df.groupby('section')['score_pct'].mean()", df)
    assert result.ok, result.error
    assert list(result.table.columns) == ["section", "score_pct"]
    assert result.table["score_pct"].tolist() == [57.5, 37.5]


def test_sorted_frame_has_no_leftover_index_column(df):
    result = execute("python", "result = df.sort_values('score_pct')[['student_id', 'score_pct']]", df)
    assert result.ok
    assert list(result.table.columns) == ["student_id", "score_pct"]
    assert result.table["student_id"].tolist() == ["S004", "S002", "S003", "S001"]


def test_meaningful_index_is_kept(df):
    result = execute("python", "result = df['section'].value_counts()", df)
    assert list(result.table.columns) == ["section", "count"]
    result = execute("python", "result = df.groupby(['section', 'result']).size()", df)
    assert list(result.table.columns) == ["section", "result", "value"]


def test_scalar_result_and_trailing_expression(df):
    result = execute("python", "x = (df['result'] == 'Fail').mean() * 100\nx", df)
    assert result.ok and result.kind == "scalar"
    assert result.scalar == 50.0


def test_figure_is_returned_as_json(df):
    code = "result = df\nfig = px.bar(df, x='student_id', y='score_pct', title='Scores')"
    result = execute("python", code, df)
    assert result.ok, result.error
    figure = json.loads(result.figure_json)
    assert figure["layout"]["title"]["text"] == "Scores"


def test_runtime_error_reports_type_and_line(df):
    result = execute("python", "a = 1\nresult = df['no_such_column']", df)
    assert not result.ok
    assert result.error.startswith("KeyError")
    assert "line 2" in result.error


def test_input_frame_is_not_modified(df):
    before = df.copy()
    execute("python", "df['score_pct'] = 0\nresult = df", df)
    pd.testing.assert_frame_equal(df, before)


def test_select_and_with_queries_work(df):
    result = execute("sql", "SELECT section, AVG(score_pct) AS avg FROM marks GROUP BY 1 ORDER BY 1", df)
    assert result.ok, result.error
    assert result.table["avg"].tolist() == [57.5, 37.5]
    result = execute("sql", "WITH f AS (SELECT * FROM marks WHERE result = 'Fail') SELECT COUNT(*) AS n FROM f;", df)
    assert result.ok and result.table["n"].iloc[0] == 2


def test_large_results_are_capped(df):
    big = pd.DataFrame({"x": range(12_000)})
    result = execute("python", "result = df", big)
    assert result.ok
    assert result.total_rows == 12_000
    assert len(result.table) == 5000


def test_figures_use_the_app_palette_when_given(df):
    style = {"colorway": ["#2a78d6", "#eb6834"], "font_family": "system-ui"}
    result = execute("python", "result = 1\nfig = px.bar(df, x='section', y='score_pct')", df, figure_style=style)
    figure = json.loads(result.figure_json)
    assert figure["data"][0]["marker"]["color"] == "#2a78d6"
    plain = json.loads(execute("python", "result = 1\nfig = px.bar(df, x='section', y='score_pct')", df).figure_json)
    assert plain["data"][0]["marker"]["color"] != "#2a78d6"
