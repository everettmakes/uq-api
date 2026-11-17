import statistics
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from uq_api.main import app
from uq_api.model import GPSurrogate

pytestmark = pytest.mark.perf


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_predict_latency_p95(client):
    """95% of single-point predictions return within 200ms."""
    timings = []
    for _ in range(50):
        start = time.perf_counter()
        r = client.post("/predict", json={"x": [[2.5]]})
        timings.append(time.perf_counter() - start)
        assert r.status_code == 200

    p95 = statistics.quantiles(timings, n=20)[-1]
    assert p95 < 0.2, f"p95 latency {p95:.3f}s"


def test_batch_of_1000_points_is_fast():
    """Predicting 1000 points in one call takes under 1 second."""
    X = np.linspace(0, 5, 20).reshape(-1, 1)
    model = GPSurrogate()
    model.fit(X, np.sin(X).ravel())

    start = time.perf_counter()
    model.predict(np.linspace(-1, 6, 1000).reshape(-1, 1))
    assert time.perf_counter() - start < 1.0


def test_fit_on_500_points_is_fast():
    """Training on 500 points takes under 5 seconds (GP training is O(n³))."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(500, 1))
    y = np.sin(X).ravel()

    start = time.perf_counter()
    GPSurrogate().fit(X, y)
    assert time.perf_counter() - start < 5.0