import numpy as np


def test_predict_returns_one_value_per_input(trained_gp):
    X_new = np.array([[0.5], [2.5], [6.0]])
    mean, std = trained_gp.predict(X_new)
    assert mean.shape == (3,)
    assert std.shape == (3,)


def test_std_is_never_negative(trained_gp):
    X_new = np.linspace(-1, 6, 50).reshape(-1, 1)
    _, std = trained_gp.predict(X_new)
    assert (std >= 0).all()


def test_uncertainty_grows_away_from_data(trained_gp):
    _, std_near = trained_gp.predict(np.array([[0.5]]))
    _, std_gap = trained_gp.predict(np.array([[2.5]]))
    _, std_far = trained_gp.predict(np.array([[8.0]]))
    assert std_near < std_gap < std_far


def test_mean_matches_training_data(trained_gp, training_data):
    X, y = training_data
    mean, _ = trained_gp.predict(X)
    np.testing.assert_allclose(mean, y, atol=0.05)