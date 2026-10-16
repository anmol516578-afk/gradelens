"""Everything a tab needs to render, computed once per run in app.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.config import AnalysisConfig
from core.pipeline import PreparedData
from ui.sidebar import Filters


@dataclass
class AppContext:
    prepared: PreparedData
    raw: pd.DataFrame
    config: AnalysisConfig
    derived_all: pd.DataFrame
    filtered: pd.DataFrame
    filters: Filters
    section_colors: dict[str, str]
    risk: Any = None  # ml.risk_model.RiskResult once computed
    extras: dict = field(default_factory=dict)

    @property
    def semester_scope(self) -> pd.DataFrame:
        """All subjects/sections, but respecting the semester filter (for profiles)."""
        if self.filters.semester:
            return self.derived_all[self.derived_all["semester"] == self.filters.semester]
        return self.derived_all
