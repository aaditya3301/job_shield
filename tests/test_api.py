import joblib
import pytest
from fastapi.testclient import TestClient

from app import MODEL_PATH, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


POSTING = {
    "title": "Software Engineer",
    "description": "Develop Python applications with our engineering team. Full time position with a structured interview process.",
    "requirements": "Python, SQL and two years of software development experience.",
}


def test_health_and_metadata(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert len(response.json()["model_sha256"]) == 64
    assert client.get("/model-info").json()["metadata"]["scikit_learn_version"] == "1.7.2"


def test_prediction_matches_saved_model(client):
    result = client.post("/predict", json=POSTING)
    assert result.status_code == 200
    data = result.json()
    artifact = joblib.load(MODEL_PATH)
    expected = float(artifact["pipeline"].predict_proba([" ".join(POSTING.values())])[0, 1])
    assert data["fraud_probability"] == pytest.approx(expected)
    assert data["fraudulent"] == int(expected >= 0.5)
    assert data["risk_level"] == ("Low" if expected < 0.3 else "High" if expected >= 0.7 else "Medium")
    assert data["recognized_features"] > 0
    assert all(item["contribution"] > 0 for item in data["fraud_signals"])
    assert all(item["contribution"] < 0 for item in data["genuine_signals"])


@pytest.mark.parametrize("payload", [
    {}, {"description": "short"}, {"description": " " * 30},
    {"description": "1" * 30}, {"description": "x" * 30001},
    {**POSTING, "unrecognized_field": "typo"},
])
def test_invalid_input_is_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_unknown_vocabulary_is_disclosed(client):
    result = client.post("/predict", json={"description": "zxqvbnmasdf zxqvbnmasdf zxqvbnmasdf"})
    assert result.status_code == 200
    assert result.json()["recognized_features"] == 0
    assert "not informative" in result.json()["note"]
