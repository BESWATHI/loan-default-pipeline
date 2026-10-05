import numpy as np
import pandas as pd

from src.features.build import FEATURES, add_derived_features, build_preprocessor


def test_output_has_exactly_the_model_features(loans):
    out = add_derived_features(loans)
    assert list(out.columns) == FEATURES
    assert "is_default" not in out.columns  # the target never leaks into features


def test_derived_features_are_correct(loans):
    row = loans.iloc[[0]].copy()
    row[["loan_amnt", "annual_inc", "installment", "fico_range_low", "fico_range_high"]] = [10000, 50000, 300, 700, 704]
    out = add_derived_features(row).iloc[0]
    assert out["fico_avg"] == 702
    assert out["loan_to_income"] == 0.2
    assert out["installment_to_income"] == 300 * 12 / 50000


def test_zero_income_gives_missing_ratio_not_infinity(loans):
    row = loans.iloc[[0]].copy()
    row["annual_inc"] = 0
    out = add_derived_features(row).iloc[0]
    assert np.isnan(out["loan_to_income"])


def test_preprocessor_handles_missing_and_unseen_values(loans):
    prep = build_preprocessor().fit(add_derived_features(loans))
    new = loans.iloc[[0]].copy()
    new["addr_state"] = "ZZ"      # category never seen in training
    new["revol_util"] = np.nan    # missing value
    transformed = prep.transform(add_derived_features(new))
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert not np.isnan(dense).any()
