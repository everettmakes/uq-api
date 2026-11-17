import pytest
from fastapi.testclient import TestClient

from uq_api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:      # "with" runs the lifespan, so the model gets trained
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_mean_and_std(client):
    r = client.post("/predict", json={"x": [[0.5], [2.5]]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["mean"]) == 2
    assert len(body["std"]) == 2


def test_prediction_is_stored_and_retrievable(client):
    run_id = client.post("/predict", json={"x": [[0.5]]}).json()["run_id"]
    r = client.get(f"/runs/{run_id}")
    assert r.status_code == 200
    assert r.json()["x"] == [[0.5]]


def test_unknown_run_returns_404(client):
    r = client.get("/runs/does-not-exist")
    assert r.status_code == 404


def test_invalid_input_returns_422(client):
    r = client.post("/predict", json={"x": "not a list"})
    assert r.status_code == 422