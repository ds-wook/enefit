import warnings
from pathlib import Path

import pandas as pd
import polars as pl
from omegaconf import DictConfig

warnings.filterwarnings("ignore")


class DataStorage:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.df_data = pl.read_csv(
            Path(self.cfg.data.root) / "train.csv", columns=self.cfg.data.data_cols, try_parse_dates=True
        )
        self.df_client = pl.read_csv(
            Path(self.cfg.data.root) / "client.csv", columns=self.cfg.data.client_cols, try_parse_dates=True
        )
        self.df_gas_prices = pl.read_csv(
            Path(self.cfg.data.root) / "gas_prices.csv", columns=self.cfg.data.gas_prices_cols, try_parse_dates=True
        )
        self.df_electricity_prices = pl.read_csv(
            Path(self.cfg.data.root) / "electricity_prices.csv",
            columns=self.cfg.data.electricity_prices_cols,
            try_parse_dates=True,
        )
        self.df_forecast_weather = pl.read_csv(
            Path(self.cfg.data.root) / "forecast_weather.csv",
            columns=self.cfg.data.forecast_weather_cols,
            try_parse_dates=True,
        )
        self.df_historical_weather = pl.read_csv(
            Path(self.cfg.data.root) / "historical_weather.csv",
            columns=self.cfg.data.historical_weather_cols,
            try_parse_dates=True,
        )
        self.df_weather_station_to_county_mapping = pl.read_csv(
            Path(self.cfg.data.root) / "weather_station_to_county_mapping.csv",
            columns=self.cfg.data.location_cols,
            try_parse_dates=True,
        )
        self.df_data = self.df_data.filter(pl.col("datetime") >= pd.to_datetime("2022-01-01"))
        self.df_target = self.df_data.select(self.cfg.data.target_cols)

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

    def update_with_new_data(
        self,
        df_new_client,
        df_new_gas_prices,
        df_new_electricity_prices,
        df_new_forecast_weather,
        df_new_historical_weather,
        df_new_target,
    ):
        df_new_client = pl.from_pandas(df_new_client[self.cfg.client_cols], schema_overrides=self.schema_client)
        df_new_gas_prices = pl.from_pandas(
            df_new_gas_prices[self.cfg.gas_prices_cols],
            schema_overrides=self.schema_gas_prices,
        )
        df_new_electricity_prices = pl.from_pandas(
            df_new_electricity_prices[self.cfg.electricity_prices_cols],
            schema_overrides=self.schema_electricity_prices,
        )
        df_new_forecast_weather = pl.from_pandas(
            df_new_forecast_weather[self.cfg.forecast_weather_cols],
            schema_overrides=self.schema_forecast_weather,
        )
        df_new_historical_weather = pl.from_pandas(
            df_new_historical_weather[self.cfg.historical_weather_cols],
            schema_overrides=self.schema_historical_weather,
        )
        df_new_target = pl.from_pandas(df_new_target[self.cfg.target_cols], schema_overrides=self.schema_target)

        self.df_client = pl.concat([self.df_client, df_new_client]).unique(
            ["date", "county", "is_business", "product_type"]
        )
        self.df_gas_prices = pl.concat([self.df_gas_prices, df_new_gas_prices]).unique(["forecast_date"])
        self.df_electricity_prices = pl.concat([self.df_electricity_prices, df_new_electricity_prices]).unique(
            ["forecast_date"]
        )
        self.df_forecast_weather = pl.concat([self.df_forecast_weather, df_new_forecast_weather]).unique(
            ["forecast_datetime", "latitude", "longitude", "hours_ahead"]
        )
        self.df_historical_weather = pl.concat([self.df_historical_weather, df_new_historical_weather]).unique(
            ["datetime", "latitude", "longitude"]
        )
        self.df_target = pl.concat([self.df_target, df_new_target]).unique(
            ["datetime", "county", "is_business", "product_type", "is_consumption"]
        )

    def preprocess_test(self, df_test):
        df_test = df_test.rename(columns={"prediction_datetime": "datetime"})
        df_test = pl.from_pandas(df_test[self.cfg.data_cols[1:]], schema_overrides=self.schema_data)
        return df_test
