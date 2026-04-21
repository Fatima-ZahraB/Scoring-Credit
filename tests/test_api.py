import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Dummy model (GLOBAL) ──────────────────────────────────────────

class DummyModel:
    def __init__(self, proba=0.3):
        self.proba = proba

    def predict_proba(self, X):
        return np.array([[1 - self.proba, self.proba]])


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_predictor_loaded():
    mock_pred = MagicMock()
    mock_pred.model_loaded = True
    mock_pred.data_loaded = True
    mock_pred.total_clients = 100
    mock_pred.predict.return_value = (0.15, 3.2)

    with patch("app.main.predictor", mock_pred):
        yield mock_pred


@pytest.fixture
def client(mock_predictor_loaded):
    from app.main import app
    return TestClient(app)


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_fields(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "data_loaded" in data
        assert "total_clients" in data

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["data_loaded"] is True


# ── Predict endpoint ──────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_success(self, client):
        resp = client.post("/predict", json={"SK_ID_CURR": 100001})
        assert resp.status_code == 200

    def test_predict_response_fields(self, client):
        data = client.post("/predict", json={"SK_ID_CURR": 100001}).json()
        assert "probability_default" in data
        assert "risk_level" in data
        assert "recommendation" in data
        assert "inference_time_ms" in data
        assert "timestamp" in data

    def test_predict_probability_range(self, client):
        data = client.post("/predict", json={"SK_ID_CURR": 100001}).json()
        assert 0.0 <= data["probability_default"] <= 1.0

    def test_predict_low_risk_label(self, client, mock_predictor_loaded):
        mock_predictor_loaded.predict.return_value = (0.10, 2.0)
        data = client.post("/predict", json={"SK_ID_CURR": 100001}).json()
        assert data["risk_level"] == "LOW"

    def test_predict_medium_risk_label(self, client, mock_predictor_loaded):
        mock_predictor_loaded.predict.return_value = (0.45, 2.0)
        data = client.post("/predict", json={"SK_ID_CURR": 100001}).json()
        assert data["risk_level"] == "MEDIUM"

    def test_predict_high_risk_label(self, client, mock_predictor_loaded):
        mock_predictor_loaded.predict.return_value = (0.75, 2.0)
        data = client.post("/predict", json={"SK_ID_CURR": 100001}).json()
        assert data["risk_level"] == "HIGH"

    def test_predict_client_not_found(self, client, mock_predictor_loaded):
        mock_predictor_loaded.predict.side_effect = ValueError("SK_ID_CURR 999 not found")
        resp = client.post("/predict", json={"SK_ID_CURR": 999})
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_predict_invalid_id_zero(self, client):
        resp = client.post("/predict", json={"SK_ID_CURR": 0})
        assert resp.status_code == 422

    def test_predict_invalid_id_negative(self, client):
        resp = client.post("/predict", json={"SK_ID_CURR": -1})
        assert resp.status_code == 422

    def test_predict_missing_body(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_predict_string_id(self, client):
        resp = client.post("/predict", json={"SK_ID_CURR": "abc"})
        assert resp.status_code == 422


# ── Predictor unit tests ──────────────────────────────────────────────────────

class TestPredictor:
    def test_predict_raises_if_not_loaded(self):
        from app.predictor import Predictor
        p = Predictor()
        with pytest.raises(RuntimeError, match="not initialized"):
            p.predict(100001)

    def test_total_clients_zero_before_load(self):
        from app.predictor import Predictor
        p = Predictor()
        assert p.total_clients == 0

    def test_predict_client_not_found(self, tmp_path):
        from app.predictor import Predictor
        import joblib

        model_path = tmp_path / "model.pkl"
        joblib.dump(DummyModel(0.15), model_path)

        data = pd.DataFrame({
            "SK_ID_CURR": [100001, 100002],
            "feature_a": [1.0, 2.0],
        })

        data_path = tmp_path / "clients.csv"
        data.to_csv(data_path, index=False)

        p = Predictor()
        with patch("app.predictor.MODEL_PATH", model_path), \
             patch("app.predictor.DATA_PATH", data_path):
            p.load()
            with pytest.raises(ValueError, match="not found"):
                p.predict(999999)

    def test_predict_returns_correct_proba(self, tmp_path):
        from app.predictor import Predictor
        import joblib

        model_path = tmp_path / "model.pkl"
        joblib.dump(DummyModel(0.30), model_path)

        data = pd.DataFrame({
            "SK_ID_CURR": [100001],
            "feature_a": [1.0],
        })

        data_path = tmp_path / "clients.csv"
        data.to_csv(data_path, index=False)

        p = Predictor()
        with patch("app.predictor.MODEL_PATH", model_path), \
             patch("app.predictor.DATA_PATH", data_path):
            p.load()
            proba, ms = p.predict(100001)
            assert abs(proba - 0.30) < 1e-6
            assert ms > 0


# ── Schema tests ──────────────────────────────────────────────────────────────

class TestSchemas:
    def test_prediction_response_low_risk(self):
        from app.schemas import PredictionResponse
        r = PredictionResponse.from_proba(sk_id=1, proba=0.1, inference_time_ms=5.0)
        assert r.risk_level == "LOW"

    def test_prediction_response_medium_risk(self):
        from app.schemas import PredictionResponse
        r = PredictionResponse.from_proba(sk_id=1, proba=0.45, inference_time_ms=5.0)
        assert r.risk_level == "MEDIUM"

    def test_prediction_response_high_risk(self):
        from app.schemas import PredictionResponse
        r = PredictionResponse.from_proba(sk_id=1, proba=0.7, inference_time_ms=5.0)
        assert r.risk_level == "HIGH"

    def test_prediction_request_valid(self):
        from app.schemas import PredictionRequest
        r = PredictionRequest(SK_ID_CURR=100001)
        assert r.SK_ID_CURR == 100001

    def test_prediction_request_invalid_zero(self):
        from pydantic import ValidationError
        from app.schemas import PredictionRequest
        with pytest.raises(ValidationError):
            PredictionRequest(SK_ID_CURR=0)