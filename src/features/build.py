"""Feature definitions shared by training and the API.

Keeping this in one place prevents training/serving skew: the model sees the
exact same transformations whether it is being trained or scoring a new loan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "is_default"

# Columns that come straight from analytics.fct_loan_features.
BASE_NUMERIC = [
    "loan_amnt",
    "term_months",
    "int_rate",
    "installment",
    "emp_length_years",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "credit_history_years",
]

CATEGORICAL_FEATURES = [
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "application_type",
]

DERIVED_FEATURES = ["fico_avg", "loan_to_income", "installment_to_income"]

# FICO low/high are replaced by their average to avoid two near-identical columns.
NUMERIC_FEATURES = [
    c for c in BASE_NUMERIC if c not in ("fico_range_low", "fico_range_high")
] + DERIVED_FEATURES

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Every raw column a caller must supply.
INPUT_COLUMNS = BASE_NUMERIC + CATEGORICAL_FEATURES


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model feature frame (FEATURES columns) from raw loan columns."""
    out = df.copy()
    for col in BASE_NUMERIC:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype("object").where(out[col].notna(), np.nan)

    income = out["annual_inc"].where(out["annual_inc"] > 0)
    out["fico_avg"] = (out["fico_range_low"] + out["fico_range_high"]) / 2
    out["loan_to_income"] = out["loan_amnt"] / income
    out["installment_to_income"] = out["installment"] * 12 / income
    return out[FEATURES]


def build_preprocessor() -> ColumnTransformer:
    """Impute and scale numeric columns; impute and one-hot encode categoricals."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
