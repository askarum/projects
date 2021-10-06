
import numpy as np

class KNN():
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train

    def predict(self, X_new):
        y_pred = []
        for x in X_new:
            dist = self.X  - x
            dist_sq = dist**2
            euclidean_distance = dist_sq.sum(axis = 1)

            match  = list(zip(euclidean_distance, self.y))
            match_k = [pair[1] for pair in sorted(match, key = lambda x: x[0])][:self.k]
            y_pred.append(max(set(match_k), key = match_k.count))
        
        return np.array(y_pred)
