"""One-call ingestion: raw sheet -> mapping -> long format -> clean data."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.cleaning import DataQualityReport, clean, infer_max_marks
from core.column_mapping import MappingEntry, guess_mapping, max_hints, normalize_to_long
from core.config import AnalysisConfig


@dataclass
class PreparedData:
    clean: pd.DataFrame
    report: DataQualityReport
    entries: list[MappingEntry]
    data_format: str
    notes: list[str] = field(default_factory=list)


def suggest_config(
    raw: pd.DataFrame, entries: list[MappingEntry], config: AnalysisConfig
) -> tuple[AnalysisConfig, dict]:
    """Adjust max marks from header hints and the data itself."""
    long_raw, _ = normalize_to_long(raw, entries)
    overrides = infer_max_marks(long_raw, config, max_hints(entries))
    return config.with_max_marks(overrides), overrides


def prepare(raw: pd.DataFrame, entries: list[MappingEntry], config: AnalysisConfig, notes=None) -> PreparedData:
    long_raw, data_format = normalize_to_long(raw, entries)
    cleaned, report = clean(long_raw, config, data_format=data_format)
    return PreparedData(cleaned, report, entries, data_format, list(notes or []))


def auto_prepare(raw: pd.DataFrame, config: AnalysisConfig) -> tuple[PreparedData, AnalysisConfig, dict]:
    """Guess everything: mapping, marks scheme, then clean."""
    guess = guess_mapping(raw)
    tuned, overrides = suggest_config(raw, guess.entries, config)
    return prepare(raw, guess.entries, tuned, guess.notes), tuned, overrides
