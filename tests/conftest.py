import numpy as np
import pytest

from uq_api.model import GPSurrogate


@pytest.fixture
def training_data():
    X = np.array([0, 0.5, 1, 1.5, 3.5, 4, 4.5, 5]).reshape(-1, 1)
    y = np.sin(X).ravel()
    return X, y


@pytest.fixture
def trained_gp(training_data):
    X, y = training_data
    model = GPSurrogate()
    model.fit(X, y)
    return model
