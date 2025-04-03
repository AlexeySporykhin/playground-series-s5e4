from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.encoding_map_ = {}

    def fit(self, X, y):
        X, y = pd.Series(X), pd.Series(y)
        self.encoding_map_ = {}
        global_mean = y.mean()

        category_stats = y.groupby(X).agg(['count', 'mean'])
        counts, means = category_stats['count'], category_stats['mean']

        # Compute smoothed means
        smoothing_factor = 1 / (1 + np.exp(-(counts - self.smoothing)))
        self.encoding_map_ = global_mean * (1 - smoothing_factor) + means * smoothing_factor

        return self

    def transform(self, X):
        X = pd.Series(X)
        return X.map(self.encoding_map_).fillna(self.encoding_map_.mean()).values.reshape(-1, 1)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


