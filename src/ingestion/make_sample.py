"""Create a smaller random sample of the full file for fast development runs.

Usage:
    python -m src.ingestion.make_sample --rows 100000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ingestion.load_raw import read_raw_chunks

DEFAULT_SOURCE = Path("data/raw/accepted_2007_to_2018Q4.csv.gz")
DEFAULT_OUTPUT = Path("data/raw/sample_loans.csv")


def make_sample(source: Path, output: Path, rows: int, seed: int = 42) -> int:
    """Sample `rows` loans spread across the whole file (not just the oldest)."""
    frames = [chunk for chunk in read_raw_chunks(source, chunksize=200_000)]
    full = pd.concat(frames, ignore_index=True)
    sample = full.sample(n=min(rows, len(full)), random_state=seed)
    sample = sample.drop(columns=["_ingested_at", "_source_file"])
    sample.to_csv(output, index=False)
    return len(sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows", type=int, default=100_000)
    args = parser.parse_args()
    n = make_sample(args.source, args.output, args.rows)
    print(f"Wrote {n:,} rows to {args.output}")
