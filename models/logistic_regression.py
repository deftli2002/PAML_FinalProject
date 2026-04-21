import numpy as np


class LogisticRegressionClassifier:
    
    def __init__(self, lr=0.01, max_iter=1000, verbose=True):
        self.lr = lr
        self.max_iter = max_iter
        self.verbose = verbose
        self.w = None
        self.b = None
        self.cost_history = []
    
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def _compute_cost(self, y, y_hat):
        n = len(y)
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return -1/n * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    
    def train(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0
        
        for i in range(self.max_iter):
            z = np.dot(X, self.w) + self.b
            y_hat = self._sigmoid(z)
            
            cost = self._compute_cost(y, y_hat)
            self.cost_history.append(cost)
            
            dw = (1/n_samples) * np.dot(X.T, (y_hat - y))
            db = (1/n_samples) * np.sum(y_hat - y)
            
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            if self.verbose and i % 100 == 0:
                print(f"Epoch {i}: cost = {cost:.4f}")
        
        if self.verbose:
            print(f"Done. Final cost: {self.cost_history[-1]:.4f}")
        
        return self
    
    def predict_prob(self, X):
        return self._sigmoid(np.dot(X, self.w) + self.b)
    
    def predict(self, X, thresh=0.5):
        return (self.predict_prob(X) >= thresh).astype(int)
    
    def get_weights(self):
        return {'w': self.w, 'b': self.b}
    
    def set_weights(self, params):
        self.w = params['w']
        self.b = params['b']


if __name__ == "__main__":
    np.random.seed(42)
    
    X = np.random.randn(100, 5)
    true_w = np.array([1.5, -2.0, 0.5, 0.8, -1.2])
    z = np.dot(X, true_w) + 0.3
    y = (1 / (1 + np.exp(-z)) > 0.5).astype(int)
    
    X_tr, X_val = X[:80], X[80:]
    y_tr, y_val = y[:80], y[80:]
    
    model = LogisticRegressionClassifier(lr=0.1, max_iter=500)
    model.train(X_tr, y_tr)
    
    preds = model.predict(X_val)
    print(f"Accuracy: {np.mean(preds == y_val):.2%}")
