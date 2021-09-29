import numpy as np

class LinearRegression:
    def __init__(self, lr = 0.01, n_iter = 500):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = None

    def fit(self,X,y):
        m,n = X.shape
        self.w = np.zeros((n,1))
        self.b = 0

        for _ in range(self.n_iter):
            y_predicted = self.predict(X)
            dw = np.dot(X.T,(y_predicted-y))/m
            db = np.sum(y_predicted-y)/m
            self.w -= self.lr * dw
            self.b -= self.lr * db


    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def coefs_(self):
        return self.w.ravel(), self.b


        