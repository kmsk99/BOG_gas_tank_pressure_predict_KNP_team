from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class DataSelector(BaseEstimator, TransformerMixin):
    def __init__(self, origin, change):
        self.origin = origin
        self.change = change

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.rename(columns={"일시": "TIME"})
        X["TIME"] = pd.to_datetime(X["TIME"])
        X = X.set_index("TIME")
        X = X[self.origin]
        X.columns = self.change
        X["YEAR"] = X.index.year
        X["DAYOFYEAR"] = X.index.dayofyear
        X["HOUR"] = X.index.hour
        return X


class NanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, zero_cols, interpolate_cols):
        self.zero_cols = zero_cols
        self.interpolate_cols = interpolate_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.zero_cols] = X[self.zero_cols].fillna(0)
        X[self.interpolate_cols] = X[self.interpolate_cols].fillna(method="ffill")
        return X


class UnitConverter(BaseEstimator, TransformerMixin):
    def __init__(self, pressure_cols, temperature_cols):
        self.pressure_cols = pressure_cols
        self.temperature_cols = temperature_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.pressure_cols] = X[self.pressure_cols] * 0.1
        X[self.temperature_cols] = X[self.temperature_cols] + 273.15
        return X
