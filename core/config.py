"""Central configuration: the canonical schema, marks scheme and thresholds.

Everything else in the app refers to the names defined here, so a column is
called the same thing in the loader, the analytics, the agent's data
dictionary and the reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# --- Canonical schema --------------------------------------------------------

STUDENT_ROLES = ("roll_no", "student_name", "section", "semester")
MARK_COMPONENTS = ("assignment", "mst1", "mst2", "end_term")
INTERNAL_COMPONENTS = ("assignment", "mst1", "mst2")
NUMERIC_ROLES = ("attendance_pct", *MARK_COMPONENTS, "internal_total", "total")

ALL_ROLES = (
    "roll_no",
    "student_name",
    "section",
    "semester",
    "subject",
    "attendance_pct",
    "assignment",
    "mst1",
    "mst2",
    "end_term",
    "internal_total",
    "total",
    "result",
)
IGNORE = "ignore"

ROLE_LABELS = {
    "roll_no": "Roll No",
    "student_name": "Student Name",
    "section": "Section",
    "semester": "Semester",
    "subject": "Subject",
    "attendance_pct": "Attendance %",
    "assignment": "Assignment",
    "mst1": "MST-1",
    "mst2": "MST-2",
    "end_term": "End Term",
    "internal_total": "Internal Total",
    "total": "Total",
    "result": "Result",
    IGNORE: "— Ignore —",
}

# Short descriptions used in the AI data dictionary and in the UI help text.
ROLE_DESCRIPTIONS = {
    "student_id": "Pseudonymous student ID (S001, S002, ...). One student has one ID across subjects.",
    "roll_no": "University roll / enrollment number.",
    "student_name": "Student's full name.",
    "section": "Class section (A, B, C, ...).",
    "semester": "Semester number.",
    "subject": "Subject / course name.",
    "attendance_pct": "Attendance percentage in this subject (0-100).",
    "assignment": "Assignment marks.",
    "mst1": "Mid-semester test 1 (MST-1) marks.",
    "mst2": "Mid-semester test 2 (MST-2) marks.",
    "end_term": "End-term (final) exam marks.",
    "internal_total": "Internal marks total (assignment + MST-1 + MST-2 unless the sheet provided its own).",
    "total": "Total marks (internal + end term unless the sheet provided its own).",
    "score_pct": "Overall percentage for this subject (total / max total x 100; uses internals when no total exists).",
    "result": "Pass / Fail / Incomplete, computed from score_pct and the configured pass mark.",
    "result_reported": "Pass / Fail exactly as written in the uploaded sheet (may be missing).",
}


# --- Analysis configuration ---------------------------------------------------


@dataclass(frozen=True)
class AnalysisConfig:
    """Marks scheme and thresholds. Frozen so it is hashable for caching."""

    max_assignment: float = 10.0
    max_mst1: float = 15.0
    max_mst2: float = 15.0
    max_end_term: float = 60.0
    # 0 means "derive from the components above".
    max_internal_total: float = 0.0
    max_total: float = 0.0
    pass_pct: float = 40.0
    attendance_threshold: float = 75.0
    mst_change_tolerance: float = 5.0  # percentage points treated as "stable"

    def max_marks(self, component: str) -> float:
        if component == "internal_total":
            return self.internal_max
        if component == "total":
            return self.total_max
        return float(getattr(self, f"max_{component}"))

    @property
    def internal_max(self) -> float:
        if self.max_internal_total > 0:
            return float(self.max_internal_total)
        return float(self.max_assignment + self.max_mst1 + self.max_mst2)

    @property
    def total_max(self) -> float:
        if self.max_total > 0:
            return float(self.max_total)
        return float(self.internal_max + self.max_end_term)

    def with_max_marks(self, overrides: dict[str, float]) -> AnalysisConfig:
        values = asdict(self)
        for component, value in overrides.items():
            values[f"max_{component}"] = float(value)
        return AnalysisConfig(**values)

    def to_key(self) -> tuple:
        """Stable, hashable representation used as a cache key."""
        return tuple(sorted(asdict(self).items()))


DEFAULT_CONFIG = AnalysisConfig()

# "Nice" maximum marks values used when inferring a marks scheme from data.
NICE_MAX_MARKS = (5, 10, 12, 15, 20, 24, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80, 90, 100, 150, 200)
