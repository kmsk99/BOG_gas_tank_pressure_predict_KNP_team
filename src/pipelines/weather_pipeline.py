from sklearn.pipeline import Pipeline

# Import user libraries
from estimators.weather_estimator import DataSelector, NanImputer, UnitConverter


def get_weather_pipeline():

    # weather columns
    origin_columns = [
        "기온(°C)",
        "지면온도(°C)",
        "현지기압(hPa)",
        "해면기압(hPa)",
        "일조(hr)",
        "풍속(m/s)",
    ]
    weather_cols = [
        "Temperature",
        "Ground_temperature",
        "Local_atmospheric_pressure",
        "Sea_level_pressure",
        "Sunshine",
        "Wind",
    ]

    zero_cols = ["Sunshine"]
    interpolate_cols = [
        "Temperature",
        "Ground_temperature",
        "Local_atmospheric_pressure",
        "Sea_level_pressure",
        "Wind",
    ]
    pressure_cols = ["Local_atmospheric_pressure", "Sea_level_pressure"]
    temperature_cols = ["Temperature", "Ground_temperature"]

    weather_pipeline = Pipeline(
        [
            ("selector", DataSelector(origin_columns, weather_cols)),
            ("imputer", NanImputer(zero_cols, interpolate_cols)),
            ("converter", UnitConverter(pressure_cols, temperature_cols)),
        ]
    )

    return weather_pipeline
