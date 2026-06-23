import numpy as np

from drybean.noise import add_feature_noise, flip_labels


def test_zero_noise_returns_equal_copy():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = add_feature_noise(x, "gaussian", 0.0, seed=42)

    assert np.array_equal(result, x)
    assert result is not x


def test_noise_does_not_mutate_original():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    original = x.copy()

    add_feature_noise(x, "missing", 0.5, seed=42)

    assert np.array_equal(x, original)


def test_label_noise_keeps_valid_class_range():
    y = np.array([0, 1, 2, 0, 1, 2])

    noisy = flip_labels(y, 0.5, n_classes=3, seed=42)

    assert set(noisy).issubset({0, 1, 2})

