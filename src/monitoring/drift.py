"""Data drift monitoring with the Population Stability Index (PSI).

PSI is the standard drift metric in credit risk. It compares how a feature is
distributed in a reference period (what the model was trained on) versus a
newer period:
    PSI < 0.10        stable
    0.10 - 0.25       moderate shift, worth watching
    > 0.25            significant shift, consider retraining

Here the reference is loans issued up to REFERENCE_MAX_YEAR and the current
period is everything after it.

Usage:
    python -m src.monitoring.drift
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build import (
    CATEGORICAL_FEATURES,
    INPUT_COLUMNS,
    NUMERIC_FEATURES,
    add_derived_features,
)

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", "reports"))
REFERENCE_MAX_YEAR = int(os.environ.get("REFERENCE_MAX_YEAR", "2015"))
EPS = 1e-6


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI using decile bins taken from the reference distribution."""
    ref, cur = reference.dropna().to_numpy(), current.dropna().to_numpy()
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (almost) constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(ref, edges)[0] / len(ref)
    cur_share = np.histogram(cur, edges)[0] / len(cur)
    return _psi(ref_share, cur_share)


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.fillna("missing").value_counts(normalize=True)
    cur = current.fillna("missing").value_counts(normalize=True)
    categories = ref.index.union(cur.index)
    return _psi(ref.reindex(categories, fill_value=0).to_numpy(), cur.reindex(categories, fill_value=0).to_numpy())


def _psi(ref_share: np.ndarray, cur_share: np.ndarray) -> float:
    ref_share = np.clip(ref_share, EPS, None)
    cur_share = np.clip(cur_share, EPS, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def classify(psi: float) -> str:
    if np.isnan(psi):
        return "no data"
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "significant"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_FEATURES:
        rows.append({"feature": col, "type": "numeric", "psi": psi_numeric(reference[col], current[col])})
    for col in CATEGORICAL_FEATURES:
        rows.append({"feature": col, "type": "categorical", "psi": psi_categorical(reference[col], current[col])})
    report = pd.DataFrame(rows)
    report["status"] = report["psi"].apply(classify)
    return report.sort_values("psi", ascending=False).reset_index(drop=True)


def write_report(report: pd.DataFrame, ref_rows: int, cur_rows: int) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORTS_DIR / "drift_report.csv", index=False)
    lines = [
        f"# Data drift report (PSI)\n",
        f"Reference: loans issued up to {REFERENCE_MAX_YEAR} ({ref_rows:,} loans). "
        f"Current: loans issued after {REFERENCE_MAX_YEAR} ({cur_rows:,} loans).\n",
        "| feature | type | PSI | status |",
        "|---|---|---|---|",
    ]
    lines += [f"| {r.feature} | {r.type} | {r.psi:.3f} | {r.status} |" for r in report.itertuples()]
    (REPORTS_DIR / "drift_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    from sqlalchemy import text

    from src.ingestion.load_raw import get_engine

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cols = ", ".join(INPUT_COLUMNS + ["issue_year"])
    with get_engine().connect() as conn:
        df = pd.read_sql(text(f"SELECT {cols} FROM analytics.fct_loan_features"), conn)

    is_reference = df["issue_year"] <= REFERENCE_MAX_YEAR
    reference = add_derived_features(df[is_reference])
    current = add_derived_features(df[~is_reference])
    report = drift_report(reference, current)
    write_report(report, len(reference), len(current))

    flagged = report[report["status"] == "significant"]["feature"].tolist()
    logger.info("Drift check done. Significant drift in: %s", ", ".join(flagged) or "none")


if __name__ == "__main__":
    main()
