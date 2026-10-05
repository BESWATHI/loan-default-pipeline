import numpy as np
import pandas as pd

from src.features.build import add_derived_features
from src.monitoring.drift import classify, drift_report, psi_categorical, psi_numeric
from tests.conftest import make_loans


def test_identical_distributions_are_stable():
    s = pd.Series(np.random.default_rng(1).normal(0, 1, 5000))
    assert psi_numeric(s, s) < 0.01


def test_shifted_distribution_is_flagged():
    rng = np.random.default_rng(2)
    ref = pd.Series(rng.normal(0, 1, 5000))
    cur = pd.Series(rng.normal(1.5, 1, 5000))
    assert classify(psi_numeric(ref, cur)) == "significant"


def test_categorical_psi_detects_new_mix():
    ref = pd.Series(["A"] * 80 + ["B"] * 20)
    cur = pd.Series(["A"] * 20 + ["B"] * 80)
    assert psi_categorical(ref, ref) < 0.01
    assert psi_categorical(ref, cur) > 0.25


def test_report_covers_every_feature():
    ref = add_derived_features(make_loans(seed=1))
    cur = add_derived_features(make_loans(seed=2))
    report = drift_report(ref, cur)
    assert set(report["feature"]) == set(ref.columns)
    assert set(report["status"]) <= {"stable", "moderate", "significant", "no data"}
