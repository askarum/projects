import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

class NaiveBayes():
    def __init__(self, X, y):
        m = X.groupby(y).mean().to_dict('index')
        self.mean = {k:list(v.values()) for k,v in m.items()}
        vs = X.groupby(y).var().to_dict('index')
        self.var = {k:list(v.values()) for k,v in vs.items()}
        self.classes = X['y'].unique()

    def calc_prior(self, X, y):
        self.prior = df.groupby('y').apply(lambda x: len(x)/len(df)).to_dict()
        return self.prior


    def calc_posterior(x):
        posterior = []
        for cl in self.classes:
            prior = self.calc_prior(X, cl)
            posterior.append(prior*multivariate_normal(x, self.mean[0][cl], np.diag(self.var[0][cl])))

        return self.classes[np.argmax(posterior)]



    def fittedvalues(self, X):
        return self.predict(X.drop(y))


    def predict(self, x):
        preds = []
        for val in x:
            preds.append(self.calc_posterior)
        return preds

