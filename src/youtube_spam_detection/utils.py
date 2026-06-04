"""Shared utilities for YouTube spam detection experiments."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

import pandas as pd


TEXT_COLUMN = "CONTENT"
LABEL_COLUMN = "CLASS"


@dataclass(frozen=True)
class DatasetColumns:
    text: str = TEXT_COLUMN
    label: str = LABEL_COLUMN


def clean_comment(text: object) -> str:
    """Normalize a YouTube comment while preserving spam-relevant tokens."""
    if pd.isna(text):
        return ""

    value = str(text).lower()
    value = re.sub(r"https?://\S+|www\.\S+", " urltoken ", value)
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def load_comment_dataset(path: str, columns: DatasetColumns | None = None) -> pd.DataFrame:
    """Load a CSV dataset and return normalized CONTENT and CLASS columns."""
    selected = columns or DatasetColumns()
    frame = pd.read_csv(path)
    resolved_text = _resolve_column(frame, selected.text)
    resolved_label = _resolve_column(frame, selected.label)

    result = frame[[resolved_text, resolved_label]].copy()
    result[resolved_text] = result[resolved_text].map(clean_comment)
    result[resolved_label] = result[resolved_label].astype(int)
    return result.rename(columns={resolved_text: TEXT_COLUMN, resolved_label: LABEL_COLUMN})


def _resolve_column(frame: pd.DataFrame, expected: str) -> str:
    """Resolve a dataset column using exact or case-insensitive matching."""
    if expected in frame.columns:
        return expected

    normalized = {str(column).strip().lower(): column for column in frame.columns}
    match = normalized.get(expected.lower())
    if match is not None:
        return str(match)

    available = ", ".join(map(str, frame.columns))
    raise ValueError(f"Missing required column '{expected}'. Available columns: {available}")
