"""Train and compare default-risk models, track them in MLflow, and save the best one.

Candidates:
- Logistic Regression with balanced class weights (interpretable baseline)
- Random Forest with balanced class weights
- XGBoost with no imbalance handling
- XGBoost with class weighting (scale_pos_weight)
- XGBoost with SMOTE oversampling

The best model by ROC-AUC on a held-out test set is saved to artifacts/model.joblib.

Usage:
    python -m src.models.train
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.features.build import (
    FEATURES,
    INPUT_COLUMNS,
    TARGET,
    add_derived_features,
    build_preprocessor,
)

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", "reports"))
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "500000"))
RANDOM_STATE = 42


@dataclass
class Candidate:
    name: str
    imbalance_strategy: str
    pipeline: object
    params: dict


def make_candidates(pos_weight: float) -> list[Candidate]:
    """Build the model pipelines to compare. pos_weight = negatives / positives."""
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    xgb_params = dict(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="auc",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf_params = dict(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=50,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    lr_params = dict(class_weight="balanced", max_iter=1000)

    def pipe(model, smote=False):
        steps = [("prep", build_preprocessor())]
        if smote:
            steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
        steps.append(("model", model))
        return Pipeline(steps)

    return [
        Candidate("logistic_regression", "class_weight", pipe(LogisticRegression(**lr_params)), lr_params),
        Candidate("random_forest", "class_weight", pipe(RandomForestClassifier(**rf_params)), rf_params),
        Candidate("xgboost", "none", pipe(XGBClassifier(**xgb_params)), xgb_params),
        Candidate(
            "xgboost_weighted",
            "scale_pos_weight",
            pipe(XGBClassifier(**xgb_params, scale_pos_weight=pos_weight)),
            {**xgb_params, "scale_pos_weight": round(pos_weight, 3)},
        ),
        Candidate("xgboost_smote", "smote", pipe(XGBClassifier(**xgb_params), smote=True), xgb_params),
    ]


def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    """Threshold-free ranking metrics plus precision/recall/F1 at a threshold.

    KS (Kolmogorov-Smirnov) is the standard separation metric in credit risk.
    """
    pred = (np.asarray(proba) >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, proba)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "ks": float(np.max(tpr - fpr)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def load_training_data(engine, sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    from sqlalchemy import text

    cols = ", ".join(INPUT_COLUMNS + [TARGET, "issue_year"])
    logger.info("Reading analytics.fct_loan_features ...")
    with engine.connect() as conn:
        df = pd.read_sql(text(f"SELECT {cols} FROM analytics.fct_loan_features"), conn)
    logger.info("Read %s finished loans (default rate %.1f%%)", f"{len(df):,}", 100 * df[TARGET].mean())

    if sample_size and len(df) > sample_size:
        df, _ = train_test_split(
            df, train_size=sample_size, stratify=df[TARGET], random_state=RANDOM_STATE
        )
        logger.info("Using a stratified sample of %s loans", f"{len(df):,}")
    return df.reset_index(drop=True)


def write_comparison(results: list[dict], best_name: str) -> pd.DataFrame:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    table.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    cols = ["model", "imbalance_strategy", "roc_auc", "pr_auc", "ks", "precision", "recall", "f1", "train_seconds"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in table.iterrows():
        cells = []
        for c in cols:
            value = row[c]
            cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        if row["model"] == best_name:
            cells[0] = f"**{cells[0]}** (best)"
        lines.append("| " + " | ".join(cells) + " |")
    (REPORTS_DIR / "model_comparison.md").write_text("\n".join(lines) + "\n")
    return table


def main() -> None:
    import mlflow

    from src.ingestion.load_raw import get_engine

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_training_data(get_engine())
    X = add_derived_features(df)
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    mlflow.set_experiment("loan-default")
    results, best = [], None

    for cand in make_candidates(pos_weight):
        logger.info("Training %s ...", cand.name)
        with mlflow.start_run(run_name=cand.name) as run:
            start = time.time()
            cand.pipeline.fit(X_train, y_train)
            seconds = time.time() - start
            proba = cand.pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, proba)

            mlflow.log_params({k: v for k, v in cand.params.items() if k != "n_jobs"})
            mlflow.log_params({"imbalance_strategy": cand.imbalance_strategy, "train_rows": len(X_train)})
            mlflow.log_metrics({**metrics, "train_seconds": seconds})

        logger.info("%s: ROC-AUC %.4f | PR-AUC %.4f | KS %.4f", cand.name, metrics["roc_auc"], metrics["pr_auc"], metrics["ks"])
        results.append({"model": cand.name, "imbalance_strategy": cand.imbalance_strategy, **metrics, "train_seconds": round(seconds, 1)})

        if best is None or metrics["roc_auc"] > best["metrics"]["roc_auc"]:
            best = {"name": cand.name, "pipeline": cand.pipeline, "metrics": metrics, "run_id": run.info.run_id}

    model_path = ARTIFACTS_DIR / "model.joblib"
    joblib.dump(best["pipeline"], model_path)

    metadata = {
        "best_model": best["name"],
        "metrics": best["metrics"],
        "features": FEATURES,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "default_rate": float(y.mean()),
    }
    (ARTIFACTS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    sample = X_test.sample(n=min(2000, len(X_test)), random_state=RANDOM_STATE)
    sample.assign(**{TARGET: y_test.loc[sample.index]}).to_csv(ARTIFACTS_DIR / "explain_sample.csv", index=False)

    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("best_model", "true")
        mlflow.log_artifact(str(model_path))

    write_comparison(results, best["name"])
    logger.info("Best model: %s (ROC-AUC %.4f). Saved to %s", best["name"], best["metrics"]["roc_auc"], model_path)


if __name__ == "__main__":
    main()
