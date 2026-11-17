"""Calibration check: when the GP is wrong, it should say so.

The API's training data ends at x = 5. At x = 6.0 the prediction is noticeably
off from the true value sin(6.0), but the true value still falls inside the
±2 std band: the model reports low confidence exactly where it is least accurate.
"""

import numpy as np
import pytest

from uq_api.main import make_training_data
from uq_api.model import GPSurrogate

X_QUERY = 6.0


@pytest.fixture
def prediction_at_6():
    X, y = make_training_data()
    model = GPSurrogate()
    model.fit(X, y)
    mean, std = model.predict(np.array([[X_QUERY]]))
    return mean[0], std[0]


def test_prediction_at_6_is_reproducible(prediction_at_6):
    mean, std = prediction_at_6
    assert mean == pytest.approx(-0.635, abs=1e-3)
    assert std == pytest.approx(0.222, abs=1e-3)


def test_true_value_falls_inside_two_std(prediction_at_6):
    mean, std = prediction_at_6
    true_value = np.sin(X_QUERY)
    assert abs(mean - true_value) > 0.3  # the prediction is noticeably wrong...
    assert abs(mean - true_value) <= 2 * std  # ...but the band covers the truth
