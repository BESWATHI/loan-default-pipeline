import pandas as pd
import pytest

from src.ingestion.load_raw import (
    METADATA_COLUMNS,
    RAW_COLUMNS,
    SourceValidationError,
    count_rows,
    load_csv,
    read_raw_chunks,
)


def _write_source(path, n_loans=5, with_footer=True, drop_column=None):
    rows = [{col: f"{col}_{i}" for col in RAW_COLUMNS} for i in range(n_loans)]
    df = pd.DataFrame(rows)
    df["total_pymnt"] = "999"  # post-loan column that must NOT be loaded
    if with_footer:
        footer = {col: None for col in df.columns}
        footer["id"] = "Total amount funded in policy code 1: 123"
        df = pd.concat([df, pd.DataFrame([footer])], ignore_index=True)
    if drop_column:
        df = df.drop(columns=[drop_column])
    df.to_csv(path, index=False)
    return path


def test_reads_only_expected_columns_and_drops_footer(tmp_path):
    source = _write_source(tmp_path / "loans.csv", n_loans=5)
    chunks = list(read_raw_chunks(source, chunksize=2))
    result = pd.concat(chunks)

    assert len(result) == 5
    assert list(result.columns) == RAW_COLUMNS + METADATA_COLUMNS
    assert "total_pymnt" not in result.columns


def test_missing_column_fails_fast(tmp_path):
    source = _write_source(tmp_path / "loans.csv", drop_column="loan_status")
    with pytest.raises(SourceValidationError, match="loan_status"):
        list(read_raw_chunks(source))


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(read_raw_chunks(tmp_path / "nope.csv"))


def test_load_is_full_refresh(tmp_path):
    sqlalchemy = pytest.importorskip("sqlalchemy")
    engine = sqlalchemy.create_engine(f"sqlite:///{tmp_path / 'warehouse.db'}")
    source = _write_source(tmp_path / "loans.csv", n_loans=7)

    # SQLite has no schemas, so the test uses schema=None.
    assert load_csv(source, engine, schema=None, chunksize=3) == 7
    assert load_csv(source, engine, schema=None, chunksize=3) == 7  # re-run
    assert count_rows(engine, schema=None) == 7  # no duplicates after re-run
