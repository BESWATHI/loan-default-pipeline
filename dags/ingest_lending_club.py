"""Airflow DAG: load the raw Lending Club file into the warehouse.

Flow: check the source file -> load it into raw.lending_club_loans -> validate the load.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task

MIN_EXPECTED_ROWS = int(os.environ.get("MIN_EXPECTED_ROWS", "1000"))


@dag(
    dag_id="ingest_lending_club",
    schedule=None,  # triggered manually; the source is a static public file
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ingestion", "lending_club"],
)
def ingest_lending_club():
    @task
    def check_source_file() -> str:
        path = Path(os.environ["RAW_DATA_PATH"])
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download the Lending Club file into data/raw/ first."
            )
        from src.ingestion.load_raw import validate_header

        validate_header(path)
        return str(path)

    @task
    def load_raw(path: str) -> int:
        from src.ingestion.load_raw import get_engine, load_csv

        return load_csv(path, get_engine())

    @task
    def validate_load(rows_loaded: int) -> None:
        from src.ingestion.load_raw import count_rows, get_engine

        rows_in_table = count_rows(get_engine())
        if rows_in_table != rows_loaded:
            raise ValueError(f"Row mismatch: loaded {rows_loaded}, table has {rows_in_table}")
        if rows_in_table < MIN_EXPECTED_ROWS:
            raise ValueError(f"Only {rows_in_table} rows loaded; expected >= {MIN_EXPECTED_ROWS}")

    validate_load(load_raw(check_source_file()))


ingest_lending_club()
