import numpy as np

class LogisticRegression():

    def __init__(self, lr = 0.05, n_iters = 100):
        self.w = None
        self.b = None
        self.lr = lr
        self.n_iters = n_iters

    
    def fit(self, X, y):
        X = X.values
        y = y.values
        n, m = X.shape[0], X.shape[1]
        self.w = np.zeros(m)
        self.b = 0

        for i in range(self.n_iters):
            preds = self.predict(X, prob = True)

            dw = 1/n * np.dot(X.T, (preds - y))
            db = 1/n * np.sum(preds - y)

            self.w = self.w - dw*self.lr
            self.b = self.b - db*self.b

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))


    def predict(self, X, prob = False):
        linear = np.dot(X, self.w) + self.b
        predictions = self.sigmoid(linear)
        if not prob:
            predictions = np.where(predictions <= 0.5, 0, 1)

        return predictions