"""Download and prepare the KaggleHub YouTube comments spam dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import kagglehub
import pandas as pd


DATASET_HANDLE = "ahsenwaheed/youtube-comments-spam-dataset"
REQUIRED_COLUMNS = {"CONTENT", "CLASS"}


def find_valid_csv_files(dataset_path: Path) -> list[Path]:
    """Return CSV files that contain the expected YouTube spam columns."""
    matches: list[Path] = []
    for csv_path in dataset_path.rglob("*.csv"):
        try:
            header = pd.read_csv(csv_path, nrows=0)
        except Exception:
            continue

        normalized = {str(column).strip().upper() for column in header.columns}
        if REQUIRED_COLUMNS.issubset(normalized):
            matches.append(csv_path)

    return matches


def read_spam_csv(path: Path) -> pd.DataFrame:
    """Read one source CSV and normalize the required column names."""
    frame = pd.read_csv(path)
    rename_map = {column: str(column).strip().upper() for column in frame.columns}
    frame = frame.rename(columns=rename_map)

    selected = frame[["CONTENT", "CLASS"]].copy()
    selected["SOURCE_FILE"] = path.name
    return selected


def download_dataset(output: Path) -> Path:
    """Download the KaggleHub dataset and write one combined CSV."""
    dataset_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    csv_files = find_valid_csv_files(dataset_path)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files with CONTENT and CLASS found in {dataset_path}")

    combined = pd.concat([read_spam_csv(path) for path in csv_files], ignore_index=True)
    combined = combined.dropna(subset=["CONTENT", "CLASS"])
    combined["CLASS"] = combined["CLASS"].astype(int)

    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the YouTube comments spam dataset from KaggleHub.")
    parser.add_argument(
        "--output",
        default="data/youtube_comments_spam.csv",
        type=Path,
        help="Output CSV path for the combined dataset.",
    )
    return parser.parse_args()


def main() -> None:
    output = download_dataset(parse_args().output)
    print(f"Prepared dataset: {output}")


if __name__ == "__main__":
    main()
