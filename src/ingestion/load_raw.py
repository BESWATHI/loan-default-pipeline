"""Load the raw Lending Club loan file into the warehouse `raw` schema.

Design choices:
- Every column is loaded as text. Type casting happens later in dbt, so the raw
  layer is an exact copy of the source and bad values never break the load.
- Only columns known at loan application time are kept. Payment fields such as
  `total_pymnt` or `recoveries` are only known after a loan ends, so using them
  would leak the answer into the model.
- The file is read in chunks so the ~2.2M-row file never has to fit in memory.
- Each run is a full refresh, which keeps re-runs safe and repeatable.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)

RAW_SCHEMA = "raw"
RAW_TABLE = "lending_club_loans"

# Application-time fields plus the target column (loan_status).
RAW_COLUMNS = [
    "id",
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "issue_d",
    "loan_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "application_type",
    "mort_acc",
    "pub_rec_bankruptcies",
]

METADATA_COLUMNS = ["_ingested_at", "_source_file"]


class SourceValidationError(ValueError):
    """Raised when the source file is missing expected columns."""


def validate_header(path: Path) -> None:
    """Fail fast if the file is missing any column we depend on."""
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(RAW_COLUMNS) - set(header))
    if missing:
        raise SourceValidationError(f"{path.name} is missing columns: {missing}")


def read_raw_chunks(path: str | Path, chunksize: int = 100_000) -> Iterator[pd.DataFrame]:
    """Yield cleaned chunks of the source file with ingestion metadata added."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    validate_header(path)

    ingested_at = datetime.now(timezone.utc).isoformat()
    reader = pd.read_csv(path, usecols=RAW_COLUMNS, dtype=str, chunksize=chunksize)

    for chunk in reader:
        # The Kaggle file ends with summary rows that are not loans; they have
        # no loan amount, so they are dropped here.
        chunk = chunk[chunk["loan_amnt"].notna()].copy()
        chunk = chunk[RAW_COLUMNS]
        chunk["_ingested_at"] = ingested_at
        chunk["_source_file"] = path.name
        yield chunk


def _qualified(schema: str | None, table: str) -> str:
    return f"{schema}.{table}" if schema else table


def load_csv(
    path: str | Path,
    engine,
    schema: str | None = RAW_SCHEMA,
    table: str = RAW_TABLE,
    chunksize: int = 100_000,
) -> int:
    """Full-refresh load of the source file. Returns the number of rows loaded."""
    from sqlalchemy import text

    with engine.begin() as conn:
        if schema:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        conn.execute(text(f"DROP TABLE IF EXISTS {_qualified(schema, table)}"))

    total = 0
    for chunk in read_raw_chunks(path, chunksize=chunksize):
        chunk.to_sql(
            table,
            engine,
            schema=schema,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1_000,  # keeps each INSERT under Postgres' parameter limit
        )
        total += len(chunk)
        logger.info("Loaded %s rows so far", f"{total:,}")

    logger.info("Finished loading %s rows into %s", f"{total:,}", _qualified(schema, table))
    return total


def count_rows(engine, schema: str | None = RAW_SCHEMA, table: str = RAW_TABLE) -> int:
    """Return the row count of the loaded table."""
    from sqlalchemy import text

    with engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {_qualified(schema, table)}")).scalar_one()


def get_engine(url: str | None = None):
    """Create a SQLAlchemy engine from WAREHOUSE_URL."""
    from sqlalchemy import create_engine

    url = url or os.environ["WAREHOUSE_URL"]
    return create_engine(url)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = load_csv(os.environ["RAW_DATA_PATH"], get_engine())
    print(f"Loaded {rows:,} rows")
