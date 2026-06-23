import numpy as np


class HandmadeGaussianNB:
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("x 必须为二维数组且与 y 样本数一致")
        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_prior_ = counts / counts.sum()
        self.theta_ = np.vstack(
            [x[y == label].mean(axis=0) for label in self.classes_]
        )
        variances = np.vstack(
            [x[y == label].var(axis=0) for label in self.classes_]
        )
        epsilon = max(
            self.var_smoothing * float(np.var(x, axis=0).max()),
            np.finfo(float).eps,
        )
        self.var_ = variances + epsilon
        return self

    def _joint_log_likelihood(self, x):
        if not hasattr(self, "classes_"):
            raise RuntimeError("模型尚未拟合")
        x = np.asarray(x, dtype=float)
        results = []
        for index in range(len(self.classes_)):
            prior = np.log(self.class_prior_[index])
            normalization = -0.5 * np.sum(
                np.log(2.0 * np.pi * self.var_[index])
            )
            distance = -0.5 * np.sum(
                ((x - self.theta_[index]) ** 2) / self.var_[index],
                axis=1,
            )
            results.append(prior + normalization + distance)
        return np.column_stack(results)

    def predict(self, x):
        scores = self._joint_log_likelihood(x)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, x):
        scores = self._joint_log_likelihood(x)
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

