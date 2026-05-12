"""
Linear regression baseline.

Trained with mean squared error against the binary label using the closed-form
ridge solution. The continuous output is used as a ranking score for the same
PR-AUC / ROC-AUC / Recall@Top-K evaluation.
"""

import numpy as np


class LinearRegressionBaseline:

    def __init__(self, l2=1.0, standardize=True):
        self.l2 = l2
        self.standardize = standardize
        self.w = None
        self.b = None
        self.mu = None
        self.sigma = None

    def _scale(self, X, fit=False):
        if not self.standardize:
            return X
        if fit:
            self.mu = X.mean(axis=0)
            self.sigma = X.std(axis=0)
            self.sigma[self.sigma == 0] = 1.0
        return (X - self.mu) / self.sigma

    def train(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        X = self._scale(X, fit=True)
        n, d = X.shape
        Xb = np.hstack([X, np.ones((n, 1))])
        reg = self.l2 * np.eye(d + 1)
        reg[-1, -1] = 0.0
        theta = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
        self.w = theta[:-1]
        self.b = float(theta[-1])
        return self

    def predict_prob(self, X):
        X = self._scale(np.asarray(X, dtype=float), fit=False)
        scores = X @ self.w + self.b
        return np.clip(scores, 0.0, 1.0)

    def predict(self, X, thresh=0.5):
        return (self.predict_prob(X) >= thresh).astype(int)
