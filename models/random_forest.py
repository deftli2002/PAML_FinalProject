"""
Random forest.

A bagged ensemble of CART-style decision trees. Each tree is trained on a
bootstrap sample using a random subset of features at every split (Gini
impurity). Sample weights are used to address heavy class imbalance, mirroring
class_weight='balanced' in scikit-learn.
"""

import numpy as np


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class _DecisionTree:

    def __init__(self, max_depth=10, min_samples_split=10, max_features=None, rng=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng or np.random.default_rng()
        self.root = None

    def _gini(self, w_pos, w_neg):
        total = w_pos + w_neg
        if total <= 0:
            return 0.0
        p = w_pos / total
        return 1.0 - p * p - (1.0 - p) * (1.0 - p)

    def _best_split(self, X, y, sw, features):
        n = len(y)
        total_pos = float(np.sum(sw * y))
        total_neg = float(np.sum(sw * (1 - y)))
        total = total_pos + total_neg
        if total <= 0:
            return None
        parent = self._gini(total_pos, total_neg)

        best_gain = 0.0
        best = None
        for f in features:
            col = X[:, f]
            order = np.argsort(col, kind="mergesort")
            col_s = col[order]
            y_s = y[order]
            sw_s = sw[order]
            cum_pos = np.cumsum(sw_s * y_s)
            cum_neg = np.cumsum(sw_s * (1 - y_s))
            cum_w = cum_pos + cum_neg

            for i in range(n - 1):
                if col_s[i] == col_s[i + 1]:
                    continue
                wl = cum_w[i]
                wr = total - wl
                if wl <= 0 or wr <= 0:
                    continue
                lp, ln = cum_pos[i], cum_neg[i]
                rp, rn = total_pos - lp, total_neg - ln
                gini_l = self._gini(lp, ln)
                gini_r = self._gini(rp, rn)
                child = (wl * gini_l + wr * gini_r) / total
                gain = parent - child
                if gain > best_gain:
                    best_gain = gain
                    best = (f, (col_s[i] + col_s[i + 1]) / 2.0)
        return best

    def _leaf_value(self, y, sw):
        total = float(np.sum(sw))
        if total <= 0:
            return 0.5
        return float(np.sum(sw * y) / total)

    def _build(self, X, y, sw, depth):
        n = len(y)
        if (depth >= self.max_depth or n < self.min_samples_split
                or np.all(y == y[0])):
            return _Node(value=self._leaf_value(y, sw))

        n_feat = X.shape[1]
        k = self.max_features or n_feat
        k = min(k, n_feat)
        features = self.rng.choice(n_feat, size=k, replace=False)

        split = self._best_split(X, y, sw, features)
        if split is None:
            return _Node(value=self._leaf_value(y, sw))
        f, thr = split
        mask = X[:, f] <= thr
        if mask.all() or (~mask).all():
            return _Node(value=self._leaf_value(y, sw))
        left = self._build(X[mask], y[mask], sw[mask], depth + 1)
        right = self._build(X[~mask], y[~mask], sw[~mask], depth + 1)
        return _Node(feature=f, threshold=thr, left=left, right=right)

    def fit(self, X, y, sw):
        self.root = self._build(X, y, sw, 0)
        return self

    def _predict_one(self, x):
        node = self.root
        while node.value is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict_prob(self, X):
        return np.array([self._predict_one(x) for x in X])


class RandomForestClassifier:

    def __init__(
        self,
        n_estimators=100,
        max_depth=12,
        min_samples_split=10,
        max_features="sqrt",
        class_weight="balanced",
        bootstrap=True,
        random_state=42,
        verbose=True,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.class_weight = class_weight
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.verbose = verbose
        self.trees = []

    def _resolve_max_features(self, n_features):
        mf = self.max_features
        if mf == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(mf, float):
            return max(1, int(mf * n_features))
        if isinstance(mf, int):
            return max(1, mf)
        return n_features

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

    def train(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)
        max_feat = self._resolve_max_features(d)
        sw_full = self._class_weights(y)

        self.trees = []
        for t in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.integers(0, n, size=n)
                Xb, yb, swb = X[idx], y[idx], sw_full[idx]
            else:
                Xb, yb, swb = X, y, sw_full
            tree = _DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_feat,
                rng=np.random.default_rng(rng.integers(0, 2**31 - 1)),
            )
            tree.fit(Xb, yb, swb)
            self.trees.append(tree)
            if self.verbose and (t + 1) % max(1, self.n_estimators // 10) == 0:
                print(f"Trained tree {t + 1}/{self.n_estimators}")
        return self

    def predict_prob(self, X):
        X = np.asarray(X, dtype=float)
        probs = np.zeros(len(X), dtype=float)
        for tree in self.trees:
            probs += tree.predict_prob(X)
        return probs / len(self.trees)

    def predict(self, X, thresh=0.5):
        return (self.predict_prob(X) >= thresh).astype(int)
