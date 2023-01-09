import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin


class TimeIndexer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["TIME"] = pd.to_datetime(X["TIME"])
        X = X.set_index("TIME")
        X["YEAR"] = X.index.year
        X["DAYOFYEAR"] = X.index.dayofyear
        X["HOUR"] = X.index.hour
        X["MINUTE"] = X.index.minute
        return X


class SeasonFeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X["SEASON"] = "NaN"
        X["SEASON"][((X.index.month >= 3) & (X.index.month <= 5))] = "0"
        X["SEASON"][((X.index.month >= 6) & (X.index.month <= 8))] = "1"
        X["SEASON"][((X.index.month >= 9) & (X.index.month <= 11))] = "2"
        X["SEASON"][
            ((X.index.month == 12) & (X.index.month >= 1) | (X.index.month <= 2))
        ] = "3"
        X = pd.get_dummies(X, columns=["SEASON"])
        return X


class WeatherInserter(BaseEstimator, TransformerMixin):
    def __init__(self, weather):
        self.weather = weather

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.merge(
            left=X, right=self.weather, how="left", on=["YEAR", "DAYOFYEAR", "HOUR"]
        ).set_index(X.index)
        return X


class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, threhold):
        self.threhold = threhold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["PRESSURE_DIFF"] = X["Local_atmospheric_pressure"] - X["PRESSURE-S"]

        # 수정사항
        X["PRESSURE-S"].loc[abs(X["PRESSURE_DIFF"]) > self.threhold] = X[
            "Local_atmospheric_pressure"
        ]
        # X["PRESSURE-S"].loc[abs(X["PRESSURE_DIFF"]) > self.threhold] = np.nan
        # X["PRESSURE-S"] = X["PRESSURE-S"].interpolate(method="time")

        X["PRESSURE_DIFF"] = X["Local_atmospheric_pressure"] - X["PRESSURE-S"]
        return X


class DateTimeConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["DAYOFYEAR_sin"] = np.sin((X["DAYOFYEAR"] - 15) * (2 * np.pi / 365.2425))
        X["DAYOFYEAR_cos"] = np.cos((X["DAYOFYEAR"] - 15) * (2 * np.pi / 365.2425))

        X["HOUR_sin"] = np.sin((X["HOUR"] + 60 * X["MINUTE"] - 24) * (2 * np.pi / 24))
        X["HOUR_cos"] = np.cos((X["HOUR"] + 60 * X["MINUTE"] - 24) * (2 * np.pi / 24))
        return X


class ProcessFeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["PIA205B-02A_DIFF"] = X["PIA205B-02A_MAX"] - X["PIA205B-02A_MIN"]
        X["PRESSURE_MAX_DIFF"] = X["PRESSURE-S"] - X["PIA205B-02A_MAX"]
        # X["PRESSURE_MIN_DIFF"] = X["PRESSURE-S"] - X["PIA205B-02A_MIN"]
        X["TI_MEAN"] = X["TI_MEAN"] + 273.15
        X["BOG"] = X["FY_SUM"] + X["FIA_SUM"]
        X["TI_SUM"] = X["FY_SUM"] + X["LP_TOTAL"]
        X["OUTLET_SUM"] = X["TI_SUM"] + X["FIA_SUM"]
        X["TI_ACC"] = X["OUTLET_SUM"] - X["STN-MFR-S"]
        X["TI_P_MAX"] = X["TI_MEAN"] / X["PIA205B-02A_MAX"]
        # X["TI_P_MIN"] = X["TI_MEAN"] / X["PIA205B-02A_MIN"]
        X["TI_VOL_MAX"] = X["TI_P_MAX"] * X["TI_SUM"]
        # X["TI_VOL_MIN"] = X["TI_P_MIN"] * X["TI_SUM"]
        return X


class WeatherFeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X["TI_T_DIV"] = X["Temperature"] - X["TI_MEAN"]
        X["T_G_DIFF"] = X["Ground_temperature"] - X["Temperature"]
        X["CONVEC"] = X["TI_T_DIV"] * X["Wind"]

        # X["T_LAP_DI"] = X["Temperature"] / X["Local_atmospheric_pressure"]
        # X["T_SLP_DI"] = X["Temperature"] / X["Sea_level_pressure"]
        # X["T_PRE_DI"] = X["Temperature"] / X["PRESSURE-S"]
        # X["TI_G_DIV"] = X["TI_MEAN"] / X["Ground_temperature"]

        # X["LAP_MIN_DIFF"] = X["Local_atmospheric_pressure"] - X["PIA205B-02A_MIN"]
        # X["LAP_MAX_DIFF"] = X["Local_atmospheric_pressure"] - X["PIA205B-02A_MAX"]
        # X["LAP_SLP_DIFF"] = X["Local_atmospheric_pressure"] - X["Sea_level_pressure"]

        # X["SLP_MIN_DIFF"] = X["Sea_level_pressure"] - X["PIA205B-02A_MIN"]
        # X["SLP_MAX_DIFF"] = X["Sea_level_pressure"] - X["PIA205B-02A_MAX"]
        # X["SLP_PRE_DIFF"] = X["Sea_level_pressure"] - X["PRESSURE-S"]
        return X
