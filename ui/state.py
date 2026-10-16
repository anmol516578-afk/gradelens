"""Session state and cached data pipeline for the Streamlit app.

Caching layers (each keyed on its inputs, so moving a slider only redoes
the cheap step it affects):

    file bytes --read--> raw sheet --prepare(mapping, marks scheme)--> clean
    clean --derive(pass mark...)--> derived  --filters--> what you see
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from core.co_attainment import parse_question_marks
from core.column_mapping import guess_mapping, mapping_from_tuple, mapping_to_tuple
from core.config import AnalysisConfig
from core.data_loader import list_sheets, read_path, read_table
from core.derive import add_derived
from core.pipeline import PreparedData, prepare, suggest_config
from core.sample_data import SAMPLE_FILES, generate_all, write_sample_files

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "sample_data"

SAMPLE_CHOICES = {
    "long": "Long format (one row per student per subject)",
    "wide": "Wide format (one row per student)",
    "xlsx": "Excel workbook (pick a sheet)",
}

CONFIG_KEYS = {
    "max_assignment": "cfg_max_assignment",
    "max_mst1": "cfg_max_mst1",
    "max_mst2": "cfg_max_mst2",
    "max_end_term": "cfg_max_end_term",
    "max_internal_total": "cfg_max_internal_total",
    "max_total": "cfg_max_total",
    "pass_pct": "cfg_pass_pct",
    "attendance_threshold": "cfg_attendance_threshold",
}
FILTER_KEYS = ("f_semester", "f_sections", "f_subjects", "drill_section", "drill_subject")
ALL = "All"


def init_state() -> None:
    defaults = {
        "source": None,  # "sample" | "upload"
        "source_name": "",
        "source_sig": None,
        "sample_kind": "long",
        "upload_bytes": None,
        "upload_name": None,
        "mapping": None,
        "mapping_auto": None,
        "mapping_notes": [],
        "data_format": "",
        "detected_overrides": {},
        "selected_student": None,
        "chat": [],
        "agent_history": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    for field, key in CONFIG_KEYS.items():
        st.session_state.setdefault(key, float(getattr(AnalysisConfig(), field)))


def current_config() -> AnalysisConfig:
    values = {field: float(st.session_state[key]) for field, key in CONFIG_KEYS.items()}
    return AnalysisConfig(**values)


def set_config(config: AnalysisConfig) -> None:
    for field, key in CONFIG_KEYS.items():
        st.session_state[key] = float(getattr(config, field))


# --- Cached I/O ------------------------------------------------------------------


def ensure_sample_files() -> None:
    if not all((SAMPLE_DIR / name).exists() for name in SAMPLE_FILES.values()):
        write_sample_files(SAMPLE_DIR)


@st.cache_data(show_spinner=False, max_entries=6)
def cached_sample_raw(kind: str, sheet: str | None = None) -> pd.DataFrame:
    try:
        ensure_sample_files()
        path = SAMPLE_DIR / SAMPLE_FILES[kind]
        return read_path(path, sheet=sheet)
    except OSError:
        # Read-only filesystem: generate in memory instead.
        bundle = generate_all()
        return bundle.messy_wide if kind == "wide" else bundle.messy_long


@st.cache_data(show_spinner=False, max_entries=4)
def cached_sample_co() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        ensure_sample_files()
        raw = read_path(SAMPLE_DIR / SAMPLE_FILES["co_marks"])
        mapping = pd.read_csv(SAMPLE_DIR / SAMPLE_FILES["co_mapping"])
    except OSError:
        bundle = generate_all()
        raw, mapping = bundle.co_marks, bundle.co_mapping
    return raw, mapping


@st.cache_data(show_spinner=False, max_entries=8)
def cached_sheets(data: bytes, filename: str) -> list[str]:
    return list_sheets(data, filename)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_read(data: bytes, filename: str, sheet: str | None) -> pd.DataFrame:
    return read_table(data, filename, sheet)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_prepare(raw: pd.DataFrame, mapping: tuple, config_key: tuple) -> PreparedData:
    config = AnalysisConfig(**dict(config_key))
    return prepare(raw, mapping_from_tuple(mapping), config)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_derive(clean: pd.DataFrame, config_key: tuple) -> pd.DataFrame:
    return add_derived(clean, AnalysisConfig(**dict(config_key)))


@st.cache_data(show_spinner=False, max_entries=8)
def cached_question_marks(raw: pd.DataFrame) -> pd.DataFrame:
    return parse_question_marks(raw)


# --- Data source handling --------------------------------------------------------


def reset_for_new_source() -> None:
    for key in (*FILTER_KEYS, "selected_student", "profile_student", "co_mapping_df", "co_upload_sig"):
        st.session_state.pop(key, None)
    st.session_state["chat"] = []
    st.session_state["agent_history"] = []
    for key in [k for k in st.session_state if str(k).startswith(("risk_", "report_"))]:
        st.session_state.pop(key, None)


def load_sample(kind: str) -> None:
    st.session_state["source"] = "sample"
    st.session_state["sample_kind"] = kind
    st.session_state["source_name"] = f"Sample data · {SAMPLE_CHOICES[kind].split(' (')[0]}"
    st.session_state["upload_bytes"] = None
    st.session_state.pop("sheet", None)


def load_upload(data: bytes, filename: str) -> None:
    st.session_state["source"] = "upload"
    st.session_state["upload_bytes"] = data
    st.session_state["upload_name"] = filename
    st.session_state["source_name"] = filename
    st.session_state.pop("sheet", None)


def sheet_options() -> list[str]:
    source = st.session_state.get("source")
    if source == "upload" and st.session_state.get("upload_bytes"):
        return cached_sheets(st.session_state["upload_bytes"], st.session_state["upload_name"])
    if source == "sample" and st.session_state.get("sample_kind") == "xlsx":
        ensure_sample_files()
        return list_sheets((SAMPLE_DIR / SAMPLE_FILES["xlsx"]).read_bytes(), SAMPLE_FILES["xlsx"])
    return []


def default_sheet(sheets: list[str]) -> str | None:
    """Prefer a sheet named like 'marks'; otherwise the first non-readme sheet."""
    if not sheets:
        return None
    for sheet in sheets:
        if "mark" in sheet.lower():
            return sheet
    for sheet in sheets:
        if "read" not in sheet.lower() and "note" not in sheet.lower():
            return sheet
    return sheets[0]


def current_raw() -> pd.DataFrame | None:
    source = st.session_state.get("source")
    sheet = st.session_state.get("sheet")
    if source == "sample":
        kind = st.session_state["sample_kind"]
        return cached_sample_raw(kind, sheet if kind == "xlsx" else None)
    if source == "upload" and st.session_state.get("upload_bytes"):
        return cached_read(st.session_state["upload_bytes"], st.session_state["upload_name"], sheet)
    return None


def source_signature() -> tuple | None:
    source = st.session_state.get("source")
    if source is None:
        return None
    if source == "upload":
        digest = hashlib.md5(st.session_state["upload_bytes"] or b"", usedforsecurity=False).hexdigest()
        return ("upload", st.session_state["upload_name"], digest, st.session_state.get("sheet"))
    return ("sample", st.session_state["sample_kind"], st.session_state.get("sheet"))


def detect_for_new_source(raw: pd.DataFrame) -> dict:
    """Guess mapping + marks scheme when the data source changes."""
    guess = guess_mapping(raw)
    st.session_state["mapping"] = guess.as_tuple()
    st.session_state["mapping_auto"] = guess.as_tuple()
    st.session_state["mapping_notes"] = guess.notes
    st.session_state["data_format"] = guess.data_format
    base = AnalysisConfig(
        pass_pct=float(st.session_state[CONFIG_KEYS["pass_pct"]]),
        attendance_threshold=float(st.session_state[CONFIG_KEYS["attendance_threshold"]]),
    )
    try:
        tuned, overrides = suggest_config(raw, guess.entries, base)
    except ValueError:
        tuned, overrides = base, {}
    set_config(tuned)
    st.session_state["detected_overrides"] = overrides
    return overrides


def apply_mapping(entries) -> None:
    st.session_state["mapping"] = mapping_to_tuple(entries)


def config_dict(config: AnalysisConfig) -> dict:
    return asdict(config)
