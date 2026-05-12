"""
Improved logistic regression classifier.

Improvements over the original version:
- Standardizes features internally (fit on train, apply on predict)
- L2 regularization
- Class-weighted loss to handle heavy class imbalance
- Mini-batch SGD with early stopping on validation loss
- Stable sigmoid + cross-entropy
"""

import numpy as np


class LogisticRegressionClassifier:

    def __init__(
        self,
        lr=0.05,
        max_iter=2000,
        batch_size=256,
        l2=1e-3,
        class_weight="balanced",
        standardize=True,
        early_stopping_patience=50,
        tol=1e-5,
        verbose=True,
        random_state=42,
    ):
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.l2 = l2
        self.class_weight = class_weight
        self.standardize = standardize
        self.early_stopping_patience = early_stopping_patience
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

        self.w = None
        self.b = None
        self.mu = None
        self.sigma = None
        self.cost_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _scale(self, X, fit=False):
        if not self.standardize:
            return X
        if fit:
            self.mu = X.mean(axis=0)
            self.sigma = X.std(axis=0)
            self.sigma[self.sigma == 0] = 1.0
        return (X - self.mu) / self.sigma

    def _class_weights(self, y):
        if self.class_weight != "balanced":
            return np.ones_like(y, dtype=float)
        n = len(y)
        n_pos = float(y.sum())
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            return np.ones_like(y, dtype=float)
        w_pos = n / (2.0 * n_pos)
        w_neg = n / (2.0 * n_neg)
        return np.where(y == 1, w_pos, w_neg)

    def _weighted_cost(self, y, y_hat, sample_w):
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1 - eps)
        loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return float(np.sum(sample_w * loss) / np.sum(sample_w))

    def train(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        X = self._scale(X, fit=True)
        if X_val is not None:
            X_val = self._scale(np.asarray(X_val, dtype=float), fit=False)
            y_val = np.asarray(y_val, dtype=float).ravel()

        n_samples, n_features = X.shape
        self.w = rng.standard_normal(n_features) * 0.01
        self.b = 0.0
        sample_w = self._class_weights(y)

        best_val = np.inf
        best_w, best_b = self.w.copy(), self.b
        patience = 0

        for epoch in range(self.max_iter):
            idx = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch = idx[start:start + self.batch_size]
                Xb, yb, wb = X[batch], y[batch], sample_w[batch]
                z = Xb @ self.w + self.b
                y_hat = self._sigmoid(z)
                err = (y_hat - yb) * wb
                wsum = wb.sum() if wb.sum() > 0 else 1.0
                dw = (Xb.T @ err) / wsum + self.l2 * self.w
                db = err.sum() / wsum
                self.w -= self.lr * dw
                self.b -= self.lr * db

            train_cost = self._weighted_cost(y, self._sigmoid(X @ self.w + self.b), sample_w)
            self.cost_history.append(train_cost)

            if X_val is not None:
                val_w = self._class_weights(y_val)
                val_cost = self._weighted_cost(
                    y_val, self._sigmoid(X_val @ self.w + self.b), val_w
                )
                if val_cost + self.tol < best_val:
                    best_val = val_cost
                    best_w, best_b = self.w.copy(), self.b
                    patience = 0
                else:
                    patience += 1
                if self.verbose and epoch % 50 == 0:
                    print(f"Epoch {epoch}: train={train_cost:.4f} val={val_cost:.4f}")
                if patience >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"Early stop at epoch {epoch}, best val={best_val:.4f}")
                    self.w, self.b = best_w, best_b
                    break
            else:
                if self.verbose and epoch % 50 == 0:
                    print(f"Epoch {epoch}: train={train_cost:.4f}")

        if self.verbose:
            print(f"Done. Final train cost: {self.cost_history[-1]:.4f}")
        return self

    def predict_prob(self, X):
        X = self._scale(np.asarray(X, dtype=float), fit=False)
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X, thresh=0.5):
        return (self.predict_prob(X) >= thresh).astype(int)

    def get_weights(self):
        return {"w": self.w, "b": self.b, "mu": self.mu, "sigma": self.sigma}

    def set_weights(self, params):
        self.w = params["w"]
        self.b = params["b"]
        self.mu = params.get("mu")
        self.sigma = params.get("sigma")
