import os
import uuid

import numpy as np
import pytest
from pymongo import MongoClient

from uq_api.repository import MongoRunRepository

MONGO_URL = os.getenv("MONGO_URL")
pytestmark = pytest.mark.skipif(
    MONGO_URL is None, reason="needs MongoDB (set MONGO_URL)"
)


@pytest.fixture
def repo():
    client = MongoClient(MONGO_URL)
    collection = client["uq_test"][
        f"runs_{uuid.uuid4().hex}"
    ]  # fresh collection per test
    yield MongoRunRepository(collection)
    collection.drop()  # clean up afterwards
    client.close()


def test_save_then_get_round_trip(repo):
    X = np.array([[0.5], [2.5]])
    run_id = repo.save(X, np.array([1.0, 2.0]), np.array([0.1, 0.2]))
    run = repo.get(run_id)
    assert run == {"x": [[0.5], [2.5]], "mean": [1.0, 2.0], "std": [0.1, 0.2]}


def test_get_missing_returns_none(repo):
    assert repo.get("does-not-exist") is None
