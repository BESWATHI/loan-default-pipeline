"""Explain the saved model with SHAP: which features push default risk up or down.

Outputs:
- reports/shap_summary.png            beeswarm plot of the top features
- reports/shap_feature_importance.csv mean |SHAP| per feature

Usage:
    python -m src.models.explain
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.build import FEATURES

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", "reports"))
MAX_ROWS = int(os.environ.get("SHAP_ROWS", "1000"))


def positive_class_values(shap_values) -> np.ndarray:
    """Normalize SHAP output to a 2-D array for the 'default' class."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def feature_importance(values: np.ndarray, names) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": list(names), "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    from sklearn.linear_model import LogisticRegression

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = joblib.load(ARTIFACTS_DIR / "model.joblib")
    sample = pd.read_csv(ARTIFACTS_DIR / "explain_sample.csv").head(MAX_ROWS)

    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    X = prep.transform(sample[FEATURES])
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    names = prep.get_feature_names_out()

    logger.info("Computing SHAP values for %s rows with %s ...", len(X), type(model).__name__)
    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, X)
    else:
        explainer = shap.TreeExplainer(model)
    values = positive_class_values(explainer.shap_values(X))

    importance = feature_importance(values, names)
    importance.to_csv(REPORTS_DIR / "shap_feature_importance.csv", index=False)

    shap.summary_plot(values, X, feature_names=names, max_display=15, show=False)
    plt.title("What drives loan default risk (SHAP)")
    plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Top 5 risk drivers: %s", ", ".join(importance["feature"].head(5)))


if __name__ == "__main__":
    main()
