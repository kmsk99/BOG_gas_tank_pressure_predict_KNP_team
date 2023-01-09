import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Import user libraries
from utils.utils import *
from estimators.train_estimator import *


def load_train_pipeline(processed_weather=None, threhold=2.5, masking=0b11111):
    drop_cols = ["YEAR", "MINUTE", "DAYOFYEAR", "HOUR"]
    y_cols = ["PIA205B-02A_MIN", "PIA205B-02A_MAX"]

    final_pipe = ColumnTransformer(
        [
            ("y_scaler", MinMaxScaler(), y_cols),
            ("dropper", "drop", drop_cols),
        ],
        sparse_threshold=0,
        remainder=MinMaxScaler(),
        verbose_feature_names_out=False,
    )

    pipe_array = [("indexer", TimeIndexer())]

    if masking & (1 << 4) > 0:
        pipe_array.append(("inserter", WeatherInserter(processed_weather)))
    if masking & (1 << 3) > 0:
        pipe_array.append(("outliere", OutlierReplacer(threhold)))
    if masking & (1 << 2) > 0:
        pipe_array.append(("process_feature", ProcessFeatureEngineering()))
    if masking & (1 << 1) > 0:
        pipe_array.append(("weather_feature", WeatherFeatureEngineering()))
    if masking & (1 << 0) > 0:
        pipe_array.append(("datetime_converter", DateTimeConverter()))
    # if (bin(masking & (1 << 0)) > 0):
    #     pipe_array.append(('season_feature', SeasonFeatureEngineering()))

    pipe_array.append(("final_pipe", final_pipe))

    train_pipeline = Pipeline(pipe_array)

    return train_pipeline
