import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.build import add_derived_features, build_preprocessor
from src.models.explain import feature_importance, positive_class_values
from src.models.train import evaluate


def test_evaluate_perfect_and_random_scores():
    y = np.array([0, 0, 1, 1])
    perfect = evaluate(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert perfect["roc_auc"] == 1.0 and perfect["ks"] == 1.0 and perfect["f1"] == 1.0
    flat = evaluate(y, np.array([0.5, 0.5, 0.5, 0.5]))
    assert flat["roc_auc"] == 0.5


def test_baseline_pipeline_learns_signal(loans):
    X, y = add_derived_features(loans), loans["is_default"]
    pipe = Pipeline([("prep", build_preprocessor()), ("model", LogisticRegression(max_iter=1000))]).fit(X, y)
    assert evaluate(y, pipe.predict_proba(X)[:, 1])["roc_auc"] > 0.8


def test_all_candidates_fit(loans):
    pytest.importorskip("xgboost")
    pytest.importorskip("imblearn")
    from src.models.train import make_candidates

    X, y = add_derived_features(loans), loans["is_default"]
    for cand in make_candidates(pos_weight=3.0):
        if "n_estimators" in cand.params:
            cand.pipeline.set_params(model__n_estimators=10)
        cand.pipeline.fit(X, y)
        proba = cand.pipeline.predict_proba(X)[:, 1]
        assert ((proba >= 0) & (proba <= 1)).all(), cand.name


def test_shap_helpers_pick_the_default_class():
    two_class = [np.zeros((3, 2)), np.ones((3, 2))]
    assert positive_class_values(two_class).sum() == 6
    assert positive_class_values(np.ones((3, 2, 2)) * [0, 2]).sum() == 12
    imp = feature_importance(np.array([[1.0, -3.0], [-1.0, 3.0]]), ["a", "b"])
    assert imp["feature"].tolist() == ["b", "a"]
