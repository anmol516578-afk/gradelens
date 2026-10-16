"""Read uploaded marks sheets (CSV / XLSX / XLS) into raw string DataFrames.

Everything is read as text so markers like "AB" survive until cleaning
decides what they mean. The loader also copes with title rows above the real
header (common in university result sheets) and odd CSV encodings.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from core.column_mapping import header_score

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")
MAX_HEADER_SCAN_ROWS = 15


class LoadError(ValueError):
    """Raised with a user-friendly message when a file cannot be read."""


def _extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise LoadError(f"Unsupported file type '{ext or filename}'. Please upload a .csv, .xlsx or .xls file.")
    return ext


def _excel_engine(ext: str) -> str:
    return "xlrd" if ext == ".xls" else "openpyxl"


def list_sheets(data: bytes, filename: str) -> list[str]:
    """Sheet names for Excel files; a single pseudo-sheet for CSV."""
    ext = _extension(filename)
    if ext == ".csv":
        return ["(CSV)"]
    try:
        with pd.ExcelFile(io.BytesIO(data), engine=_excel_engine(ext)) as book:
            return [str(name) for name in book.sheet_names]
    except Exception as exc:  # corrupted or password-protected workbook
        raise LoadError(f"Could not open the Excel file: {exc}") from exc


def _read_csv_raw(data: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        frame = pd.read_csv(io.StringIO(text), header=None, dtype=str, keep_default_na=False, skip_blank_lines=False)
        if frame.shape[1] == 1 and any(sep in text[:2000] for sep in (";", "\t", "|")):
            frame = pd.read_csv(
                io.StringIO(text),
                header=None,
                dtype=str,
                keep_default_na=False,
                sep=None,
                engine="python",
                skip_blank_lines=False,
            )
        return frame
    raise LoadError(f"Could not decode the CSV file: {last_error}")


def _read_excel_raw(data: bytes, ext: str, sheet: str | None) -> pd.DataFrame:
    try:
        frame = pd.read_excel(
            io.BytesIO(data), sheet_name=sheet or 0, header=None, dtype=str, engine=_excel_engine(ext)
        )
    except ValueError as exc:
        raise LoadError(f"Could not read sheet '{sheet}': {exc}") from exc
    except Exception as exc:
        raise LoadError(f"Could not read the Excel file: {exc}") from exc
    return frame.fillna("")


def detect_header_row(raw: pd.DataFrame) -> int:
    """Index of the row that most looks like a header (title rows are skipped)."""
    best_row, best_score = 0, -1.0
    for idx in range(min(MAX_HEADER_SCAN_ROWS, len(raw))):
        cells = [str(v).strip() for v in raw.iloc[idx].tolist()]
        score = header_score(cells)
        if score > best_score:
            best_row, best_score = idx, score
    return best_row if best_score >= 2 else 0


def _dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i, name in enumerate(names):
        name = name.strip() or f"Column {i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


def finalize_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Promote the detected header row and drop empty rows / columns."""
    raw = raw.astype(str).apply(lambda col: col.str.strip())
    raw = raw.replace({"nan": "", "None": "", "NaT": ""})
    if raw.empty:
        raise LoadError("The file is empty.")
    header_idx = detect_header_row(raw)
    header = _dedupe([str(v) for v in raw.iloc[header_idx].tolist()])
    body = raw.iloc[header_idx + 1 :].copy()
    body.columns = header
    body = body.loc[:, [c for c in body.columns if (body[c] != "").any() or not c.startswith("Column ")]]
    body = body[(body != "").any(axis=1)].reset_index(drop=True)
    if body.empty:
        raise LoadError("No data rows were found below the header.")
    return body


def read_table(data: bytes, filename: str, sheet: str | None = None) -> pd.DataFrame:
    """Read a CSV/Excel file into a DataFrame of stripped strings."""
    ext = _extension(filename)
    raw = _read_csv_raw(data) if ext == ".csv" else _read_excel_raw(data, ext, sheet)
    return finalize_frame(raw)


def read_path(path: str | Path, sheet: str | None = None) -> pd.DataFrame:
    path = Path(path)
    return read_table(path.read_bytes(), path.name, sheet)
