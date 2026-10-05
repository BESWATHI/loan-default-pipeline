import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from src.api import main as api  # noqa: E402
from src.features.build import add_derived_features, build_preprocessor  # noqa: E402

EXAMPLE = api.LoanApplication.model_config["json_schema_extra"]["examples"][0]


@pytest.fixture
def client(tmp_path, monkeypatch, loans):
    pipe = Pipeline([("prep", build_preprocessor()), ("model", LogisticRegression(max_iter=1000))])
    pipe.fit(add_derived_features(loans), loans["is_default"])
    joblib.dump(pipe, tmp_path / "model.joblib")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.joblib"))
    api.get_model.cache_clear()
    api.get_model_name.cache_clear()
    return TestClient(api.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_available": True}


def test_predict_returns_probability_and_band(client):
    body = client.post("/predict", json=EXAMPLE).json()
    assert 0 <= body["default_probability"] <= 1
    assert body["risk_band"] in {"low", "medium", "high"}


def test_invalid_input_is_rejected(client):
    bad = {**EXAMPLE, "grade": "Z"}
    assert client.post("/predict", json=bad).status_code == 422


def test_risk_bands():
    assert api.risk_band(0.05) == "low"
    assert api.risk_band(0.2) == "medium"
    assert api.risk_band(0.6) == "high"
