import numpy as np

from drybean.models.gaussian_nb import HandmadeGaussianNB


def test_handmade_gaussian_nb_fits_and_predicts_separable_data():
    x = np.array([[0.0, 0.1], [0.2, 0.0], [5.0, 5.1], [5.2, 4.9]])
    y = np.array(["A", "A", "B", "B"])

    model = HandmadeGaussianNB(var_smoothing=1e-9).fit(x, y)

    assert model.predict(np.array([[0.1, 0.1], [5.1, 5.0]])).tolist() == [
        "A",
        "B",
    ]
    probabilities = model.predict_proba(np.array([[0.1, 0.1]]))
    assert probabilities.shape == (1, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)

