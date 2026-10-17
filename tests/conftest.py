from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import AnalysisConfig  # noqa: E402
from core.derive import add_derived  # noqa: E402
from core.pipeline import auto_prepare  # noqa: E402
from core.sample_data import generate_all  # noqa: E402


@pytest.fixture(scope="session")
def bundle():
    return generate_all(seed=42)


@pytest.fixture(scope="session")
def config() -> AnalysisConfig:
    return AnalysisConfig()


@pytest.fixture(scope="session")
def prepared_long(bundle, config):
    prepared, tuned, _ = auto_prepare(bundle.messy_long, config)
    return prepared, tuned


@pytest.fixture(scope="session")
def derived(prepared_long) -> pd.DataFrame:
    prepared, tuned = prepared_long
    return add_derived(prepared.clean, tuned)


def make_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a canonical clean frame by hand (for hand-computed fixtures)."""
    df = pd.DataFrame(rows)
    for comp in ("assignment", "mst1", "mst2", "end_term"):
        if comp in df and f"{comp}_absent" not in df:
            df[f"{comp}_absent"] = False
    for text_col in ("roll_no", "student_name", "section", "semester", "subject"):
        if text_col in df:
            df[text_col] = df[text_col].astype(str)
    return df


@pytest.fixture(autouse=True, scope="session")
def isolated_model_dir(tmp_path_factory):
    """Never read or write the real models/ folder from tests."""
    import os

    path = tmp_path_factory.mktemp("models")
    previous = os.environ.get("GRADELENS_MODEL_DIR")
    os.environ["GRADELENS_MODEL_DIR"] = str(path)
    yield path
    if previous is None:
        os.environ.pop("GRADELENS_MODEL_DIR", None)
    else:
        os.environ["GRADELENS_MODEL_DIR"] = previous
