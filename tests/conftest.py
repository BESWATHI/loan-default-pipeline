import numpy as np
import pandas as pd
import pytest

from src.features.build import CATEGORICAL_FEATURES, INPUT_COLUMNS


def make_loans(n=400, seed=0):
    """Synthetic loans with a realistic shape: lower FICO and higher rate -> more defaults."""
    rng = np.random.default_rng(seed)
    fico = rng.integers(640, 820, n)
    rate = rng.uniform(5, 28, n)
    df = pd.DataFrame(
        {
            "loan_amnt": rng.uniform(1000, 40000, n),
            "term_months": rng.choice([36, 60], n),
            "int_rate": rate,
            "installment": rng.uniform(50, 1400, n),
            "emp_length_years": rng.integers(0, 11, n).astype(float),
            "annual_inc": rng.uniform(20000, 200000, n),
            "dti": rng.uniform(0, 40, n),
            "delinq_2yrs": rng.integers(0, 3, n).astype(float),
            "fico_range_low": fico,
            "fico_range_high": fico + 4,
            "inq_last_6mths": rng.integers(0, 5, n).astype(float),
            "open_acc": rng.integers(2, 30, n).astype(float),
            "pub_rec": rng.integers(0, 2, n).astype(float),
            "revol_bal": rng.uniform(0, 50000, n),
            "revol_util": rng.uniform(0, 100, n),
            "total_acc": rng.integers(5, 60, n).astype(float),
            "mort_acc": rng.integers(0, 5, n).astype(float),
            "pub_rec_bankruptcies": rng.integers(0, 2, n).astype(float),
            "credit_history_years": rng.uniform(1, 30, n),
            "grade": rng.choice(list("ABCDEFG"), n),
            "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
            "verification_status": rng.choice(["Verified", "Not Verified", "Source Verified"], n),
            "purpose": rng.choice(["debt_consolidation", "credit_card", "other"], n),
            "addr_state": rng.choice(["MA", "CA", "NY", "TX"], n),
            "application_type": rng.choice(["Individual", "Joint App"], n),
        }
    )
    risk = (rate - 15) / 5 - (fico - 720) / 40 + rng.normal(0, 1, n)
    df["is_default"] = (risk > 1.0).astype(int)
    df.loc[df.sample(frac=0.05, random_state=seed).index, "revol_util"] = np.nan  # some missing values
    assert set(INPUT_COLUMNS + CATEGORICAL_FEATURES) <= set(df.columns)
    return df


@pytest.fixture
def loans():
    return make_loans()
