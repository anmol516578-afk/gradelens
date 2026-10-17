"""Pseudonymise student identities before anything reaches an LLM.

The AI analyst only ever sees ``student_id`` values (S001, S002, ...). Real
names and roll numbers stay in the app: they are swapped back in only for
display, after the model has answered.

Masking covers every path into the model:

* the data frame the agent's code runs on (names / roll numbers dropped),
* the sample rows in the system prompt (built from that frame),
* the user's own question ("How is Aarav Sharma doing?" -> "How is S014 doing?"),
* follow-up history (stored masked).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd

ID_COLUMN = "student_id"
ID_PATTERN = re.compile(r"(?<![A-Za-z0-9_])S\d{3,}(?![A-Za-z0-9_])")
MIN_ROLL_LENGTH = 5
MIN_NAME_LETTERS = 3  # shorter roll numbers ("12") are not identifying and would collide with ordinary numbers

# Columns the agent may see, in order. Identity columns are deliberately absent.
AGENT_COLUMNS = (
    ID_COLUMN,
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
    "assignment_pct",
    "mst1_pct",
    "mst2_pct",
    "end_term_pct",
    "internal_pct",
    "score_pct",
    "result",
    "result_reported",
    "assignment_absent",
    "mst1_absent",
    "mst2_absent",
    "end_term_absent",
)


# Vowel signs, viramas and other combining marks (plus zero-width joiners) are part of a word in Indian
# scripts, but Python's \w does not include them: "राम" would match inside "श्रीराम", and "सिवा" would count
# as two letters.
_MARKS = "".join(chr(c) for c in range(0x0300, 0x2000) if unicodedata.category(chr(c)).startswith("M"))
_WORD_CHAR = "[\\w" + _MARKS + "\u200c\u200d]"


def _letter_count(text: str) -> int:
    """Letters including combining marks, so "राम" counts 3 like "Ram"."""
    return sum(unicodedata.category(ch)[0] in "LM" for ch in text)


def _phrase_pattern(phrases: list[str]) -> re.Pattern | None:
    """Case-insensitive, whitespace-tolerant, whole-word alternation (longest first)."""
    parts = []
    for phrase in sorted(set(phrases), key=len, reverse=True):
        words = phrase.split()
        if words:
            parts.append(r"\s+".join(re.escape(w) for w in words))
    if not parts:
        return None
    return re.compile(f"(?<!{_WORD_CHAR})(?:" + "|".join(parts) + f")(?!{_WORD_CHAR})", re.IGNORECASE)


@dataclass
class Pseudonymizer:
    """Stable mapping between real identities and pseudonymous IDs."""

    roll_to_id: dict[str, str]
    id_to_roll: dict[str, str]
    id_to_name: dict[str, str]
    _name_lookup: dict[str, str] = field(default_factory=dict, repr=False)
    _roll_lookup: dict[str, str] = field(default_factory=dict, repr=False)
    _name_re: re.Pattern | None = field(default=None, repr=False)
    _roll_re: re.Pattern | None = field(default=None, repr=False)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> Pseudonymizer:
        """Build from the *unfiltered* data so IDs stay stable when filters change."""
        roster = (
            df[["roll_no", "student_name"]]
            .astype(str)
            .drop_duplicates("roll_no")
            .sort_values("roll_no", kind="stable")
            .reset_index(drop=True)
        )
        width = max(3, len(str(len(roster))))
        ids = [f"S{i:0{width}d}" for i in range(1, len(roster) + 1)]
        roll_to_id = dict(zip(roster["roll_no"], ids, strict=True))
        id_to_roll = {v: k for k, v in roll_to_id.items()}
        id_to_name = dict(zip(ids, roster["student_name"], strict=True))
        pseudo = cls(roll_to_id, id_to_roll, id_to_name)
        reserved = set()
        for column in ("subject", "section", "semester"):
            if column in df:
                reserved |= {" ".join(str(v).split()).lower() for v in df[column].dropna().unique()}
        pseudo._build_patterns(reserved)
        return pseudo

    def _build_patterns(self, reserved: set[str] | None = None) -> None:
        """``reserved``: subject/section/semester values that must never be read as a name."""
        reserved = reserved or set()
        names: dict[str, str] = {}
        for sid, name in self.id_to_name.items():
            key = " ".join(str(name).split()).lower()
            # A name shared by two students is still masked; it maps to the first ID.
            # Very short names ("A") or names equal to a subject or section would corrupt ordinary
            # questions ("Section A" -> "Section S001"); they identify no one, so they are not masked.
            letters = _letter_count(key)
            if letters < MIN_NAME_LETTERS or key in reserved:
                continue
            if key and key not in ("nan", "none") and key not in names:
                names[key] = sid
        self._name_lookup = names
        self._name_re = _phrase_pattern(list(names))
        rolls = {r.lower(): sid for r, sid in self.roll_to_id.items() if len(r.strip()) >= MIN_ROLL_LENGTH}
        self._roll_lookup = rolls
        self._roll_re = _phrase_pattern(list(rolls))

    # --- masking (towards the model) ---------------------------------------------

    def mask_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace identity columns with ``student_id``; keep only analysis columns."""
        out = df.copy()
        out.insert(0, ID_COLUMN, out["roll_no"].astype(str).map(self.roll_to_id))
        columns = [c for c in AGENT_COLUMNS if c in out.columns]
        return out[columns].reset_index(drop=True)

    def mask_text(self, text: str) -> str:
        """Swap real names and (long) roll numbers in free text for IDs."""
        if not text:
            return text
        if self._name_re is not None:
            text = self._name_re.sub(lambda m: self._name_lookup[" ".join(m.group(0).split()).lower()], text)
        if self._roll_re is not None:
            text = self._roll_re.sub(lambda m: self._roll_lookup[" ".join(m.group(0).split()).lower()], text)
        return text

    # --- unmasking (towards the user) ----------------------------------------------

    def label(self, sid: str, with_roll: bool = True) -> str:
        name = self.id_to_name.get(sid)
        if name is None:
            return sid
        return f"{name} ({self.id_to_roll[sid]})" if with_roll else name

    def unmask_text(self, text: str, with_roll: bool = True) -> str:
        if not text:
            return text
        return ID_PATTERN.sub(lambda m: self.label(m.group(0), with_roll), text)

    def unmask_table(self, table: pd.DataFrame | None) -> pd.DataFrame | None:
        """Show real roll numbers and names in place of ``student_id``."""
        if table is None or table.empty:
            return table
        out = table.copy()
        if ID_COLUMN in out.columns:
            position = list(out.columns).index(ID_COLUMN)
            ids = out[ID_COLUMN].astype(str)
            out.insert(position, "student_name", ids.map(self.id_to_name).fillna(ids))
            out.insert(position, "roll_no", ids.map(self.id_to_roll).fillna(ids))
            out = out.drop(columns=[ID_COLUMN])
        for column in out.columns:
            if out[column].dtype == object or pd.api.types.is_string_dtype(out[column]):
                values = out[column]
                mask = values.notna()
                out.loc[mask, column] = values[mask].astype(str).map(lambda v: self.unmask_text(v, with_roll=False))
        return out

    def unmask_figure_json(self, figure_json: str | None) -> str | None:
        """Replace IDs in every string of a Plotly figure (axis labels, hover text, titles)."""
        if not figure_json:
            return figure_json
        data = json.loads(figure_json)

        def walk(node):
            if isinstance(node, str):
                return self.unmask_text(node, with_roll=False)
            if isinstance(node, list):
                return [walk(v) for v in node]
            if isinstance(node, dict):
                # "bdata" holds base64-encoded numeric arrays; regex replacement would corrupt them.
                return {k: (v if k == "bdata" else walk(v)) for k, v in node.items()}
            return node

        return json.dumps(walk(data))

    def real_names(self) -> list[str]:
        return [n for n in self.id_to_name.values() if n and n.lower() not in ("nan", "none")]
