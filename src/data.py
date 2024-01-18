import os
import warnings

import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")


class Warehouse:
    root = "input/predict-energy-behavior-of-prosumers"

    data_columns = [
        "target",
        "county",
        "is_business",
        "product_type",
        "is_consumption",
        "datetime",
        "row_id",
    ]
    client_columns = [
        "product_type",
        "county",
        "eic_count",
        "installed_capacity",
        "is_business",
        "date",
    ]
    gas_prices_columns = ["forecast_date", "lowest_price_per_mwh", "highest_price_per_mwh"]
    electricity_prices_columns = ["forecast_date", "euros_per_mwh"]
    forecast_weather_columns = [
        "latitude",
        "longitude",
        "hours_ahead",
        "temperature",
        "dewpoint",
        "cloudcover_high",
        "cloudcover_low",
        "cloudcover_mid",
        "cloudcover_total",
        "10_metre_u_wind_component",
        "10_metre_v_wind_component",
        "forecast_datetime",
        "direct_solar_radiation",
        "surface_solar_radiation_downwards",
        "snowfall",
        "total_precipitation",
    ]
    historical_weather_columns = [
        "datetime",
        "temperature",
        "dewpoint",
        "rain",
        "snowfall",
        "surface_pressure",
        "cloudcover_total",
        "cloudcover_low",
        "cloudcover_mid",
        "cloudcover_high",
        "windspeed_10m",
        "winddirection_10m",
        "shortwave_radiation",
        "direct_solar_radiation",
        "diffuse_radiation",
        "latitude",
        "longitude",
    ]
    location_columns = ["longitude", "latitude", "county"]
    target_columns = [
        "target",
        "county",
        "is_business",
        "product_type",
        "is_consumption",
        "datetime",
    ]

    def __init__(self):
        self.df_data = pl.read_csv(
            os.path.join(self.root, "train.csv"),
            columns=self.data_columns,
            try_parse_dates=True,
        )
        self.df_client = pl.read_csv(
            os.path.join(self.root, "client.csv"),
            columns=self.client_columns,
            try_parse_dates=True,
        )
        self.df_gas_prices = pl.read_csv(
            os.path.join(self.root, "gas_prices.csv"),
            columns=self.gas_prices_columns,
            try_parse_dates=True,
        )
        self.df_electricity_prices = pl.read_csv(
            os.path.join(self.root, "electricity_prices.csv"),
            columns=self.electricity_prices_columns,
            try_parse_dates=True,
        )
        self.df_forecast_weather = pl.read_csv(
            os.path.join(self.root, "forecast_weather.csv"),
            columns=self.forecast_weather_columns,
            try_parse_dates=True,
        )
        self.df_historical_weather = pl.read_csv(
            os.path.join(self.root, "historical_weather.csv"),
            columns=self.historical_weather_columns,
            try_parse_dates=True,
        )
        self.df_weather_station_to_county_mapping = pl.read_csv(
            os.path.join(self.root, "weather_station_to_county_mapping.csv"),
            columns=self.location_columns,
            try_parse_dates=True,
        )
        self.df_data = self.df_data.filter(pl.col("datetime") >= pd.to_datetime("2022-01-01"))
        self.df_target = self.df_data.select(self.target_columns)
        self.schema_data = self.df_data.schema
        self.schema_client = self.df_client.schema
        self.schema_gas_prices = self.df_gas_prices.schema
        self.schema_electricity_prices = self.df_electricity_prices.schema
        self.schema_forecast_weather = self.df_forecast_weather.schema
        self.schema_historical_weather = self.df_historical_weather.schema
        self.schema_target = self.df_target.schema

        self.df_weather_station_to_county_mapping = self.df_weather_station_to_county_mapping.with_columns(
            pl.col("latitude").cast(pl.datatypes.Float32),
            pl.col("longitude").cast(pl.datatypes.Float32),
        )

    def update_data(
        self,
        df_client_new,
        df_gas_price_new,
        df_elec_price_new,
        df_forecast_new,
        df_hist_weather_new,
        df_target_new,
    ):
        df_client_new = pl.from_pandas(df_client_new[self.client_columns], schema_overrides=self.schema_client)

        df_gas_price_new = pl.from_pandas(
            df_gas_price_new[self.gas_prices_columns],
            schema_overrides=self.schema_gas_prices,
        )

        df_elec_price_new = pl.from_pandas(
            df_elec_price_new[self.electricity_prices_columns],
            schema_overrides=self.schema_electricity_prices,
        )

        df_forecast_new = pl.from_pandas(
            df_forecast_new[self.forecast_weather_columns],
            schema_overrides=self.schema_forecast_weather,
        )

        df_hist_weather_new = pl.from_pandas(
            df_hist_weather_new[self.historical_weather_columns],
            schema_overrides=self.schema_historical_weather,
        )

        df_target_new = pl.from_pandas(df_target_new[self.target_columns], schema_overrides=self.schema_target)

        self.df_client = pl.concat([self.df_client, df_client_new]).unique(
            ["date", "county", "is_business", "product_type"]
        )

        self.df_gas_prices = pl.concat([self.df_gas_prices, df_gas_price_new]).unique(["forecast_date"])

        self.df_electricity_prices = pl.concat([self.df_electricity_prices, df_elec_price_new]).unique(
            ["forecast_date"]
        )

        self.df_forecast_weather = pl.concat([self.df_forecast_weather, df_forecast_new]).unique(
            ["forecast_datetime", "latitude", "longitude", "hours_ahead"]
        )

        self.df_historical_weather = pl.concat([self.df_historical_weather, df_hist_weather_new]).unique(
            ["datetime", "latitude", "longitude"]
        )

        self.df_target = pl.concat([self.df_target, df_target_new]).unique(
            ["datetime", "county", "is_business", "product_type", "is_consumption"]
        )

    def preprocess_test(self, df_test):
        df_test = df_test.rename(columns={"prediction_datetime": "datetime"})
        df_test = pl.from_pandas(df_test[self.data_columns[1:]], schema_overrides=self.schema_data)
        return df_test
