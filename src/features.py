import datetime

import holidays
import numpy as np
import pandas as pd
import polars as pl


class FeatureEngineer:
    def __init__(self, data):
        self.data = data
        self.estonian_holidays = list(holidays.country_holidays("EE", years=range(2021, 2026)).keys())

    def _general_features(self, df_features):
        df_features = (
            df_features.with_columns(
                pl.col("datetime").dt.ordinal_day().alias("dayofyear"),
                pl.col("datetime").dt.hour().alias("hour"),
                pl.col("datetime").dt.day().alias("day"),
                pl.col("datetime").dt.weekday().alias("weekday"),
                pl.col("datetime").dt.month().alias("month"),
                pl.col("datetime").dt.year().alias("year"),
            )
            .with_columns(
                pl.concat_str(
                    "county",
                    "is_business",
                    "product_type",
                    "is_consumption",
                    separator="_",
                ).alias("segment"),
            )
            .with_columns(
                (np.pi * pl.col("dayofyear") / 183).sin().alias("sin(dayofyear)"),
                (np.pi * pl.col("dayofyear") / 183).cos().alias("cos(dayofyear)"),
                (np.pi * pl.col("hour") / 12).sin().alias("sin(hour)"),
                (np.pi * pl.col("hour") / 12).cos().alias("cos(hour)"),
            )
        )
        return df_features

    def _client_features(self, df_features):
        df_client = self.data.df_client

        df_features = df_features.join(
            df_client.with_columns((pl.col("date") + pl.duration(days=2)).cast(pl.Date)),
            on=["county", "is_business", "product_type", "date"],
            how="left",
        )
        return df_features

    def is_country_holiday(self, row):
        return datetime.date(row["year"], row["month"], row["day"]) in self.estonian_holidays

    def _holidays_features(self, df_features):
        df_features = df_features.with_columns(
            pl.struct(["year", "month", "day"]).apply(self.is_country_holiday).alias("is_country_holiday")
        )
        return df_features

    def _forecast_weather_features(self, df_features):
        df_forecast_weather = self.data.df_forecast_weather
        df_weather_station_to_county_mapping = self.data.df_weather_station_to_county_mapping

        df_forecast_weather = (
            df_forecast_weather.rename({"forecast_datetime": "datetime"})
            .filter((pl.col("hours_ahead") >= 22) & pl.col("hours_ahead") <= 45)
            #             .drop("hours_ahead")
            .with_columns(
                pl.col("latitude").cast(pl.datatypes.Float32),
                pl.col("longitude").cast(pl.datatypes.Float32),
            )
            .join(
                df_weather_station_to_county_mapping,
                how="left",
                on=["longitude", "latitude"],
            )
            .drop("longitude", "latitude")
        )

        df_forecast_weather_date = df_forecast_weather.group_by("datetime").mean().drop("county")

        df_forecast_weather_local = (
            df_forecast_weather.filter(pl.col("county").is_not_null()).group_by("county", "datetime").mean()
        )

        for hours_lag in [0, 7 * 24]:
            df_features = df_features.join(
                df_forecast_weather_date.with_columns(pl.col("datetime") + pl.duration(hours=hours_lag)),
                on="datetime",
                how="left",
                suffix=f"_forecast_{hours_lag}h",
            )
            df_features = df_features.join(
                df_forecast_weather_local.with_columns(pl.col("datetime") + pl.duration(hours=hours_lag)),
                on=["county", "datetime"],
                how="left",
                suffix=f"_forecast_local_{hours_lag}h",
            )

        return df_features

    def _historical_weather_features(self, df_features):
        df_historical_weather = self.data.df_historical_weather
        df_weather_station_to_county_mapping = self.data.df_weather_station_to_county_mapping

        df_historical_weather = (
            df_historical_weather.with_columns(
                pl.col("latitude").cast(pl.datatypes.Float32),
                pl.col("longitude").cast(pl.datatypes.Float32),
            )
            .join(
                df_weather_station_to_county_mapping,
                how="left",
                on=["longitude", "latitude"],
            )
            .drop("longitude", "latitude")
        )

        df_historical_weather_date = df_historical_weather.group_by("datetime").mean().drop("county")

        df_historical_weather_local = (
            df_historical_weather.filter(pl.col("county").is_not_null()).group_by("county", "datetime").mean()
        )

        for hours_lag in [2 * 24, 7 * 24]:
            df_features = df_features.join(
                df_historical_weather_date.with_columns(pl.col("datetime") + pl.duration(hours=hours_lag)),
                on="datetime",
                how="left",
                suffix=f"_historical_{hours_lag}h",
            )
            df_features = df_features.join(
                df_historical_weather_local.with_columns(pl.col("datetime") + pl.duration(hours=hours_lag)),
                on=["county", "datetime"],
                how="left",
                suffix=f"_historical_local_{hours_lag}h",
            )

        for hours_lag in [1 * 24]:
            df_features = df_features.join(
                df_historical_weather_date.with_columns(
                    pl.col("datetime") + pl.duration(hours=hours_lag),
                    pl.col("datetime").dt.hour().alias("hour"),
                )
                .filter(pl.col("hour") <= 10)
                .drop("hour"),
                on="datetime",
                how="left",
                suffix=f"_historical_{hours_lag}h",
            )

        return df_features

    def _target_features(self, df_features):
        df_target = self.data.df_target

        df_target_all_type_sum = (
            df_target.group_by(["datetime", "county", "is_business", "is_consumption"]).sum().drop("product_type")
        )

        df_target_all_county_type_sum = (
            df_target.group_by(["datetime", "is_business", "is_consumption"]).sum().drop("product_type", "county")
        )

        hours_list = [i * 24 for i in range(2, 15)]

        for hours_lag in hours_list:
            df_features = df_features.join(
                df_target.with_columns(pl.col("datetime") + pl.duration(hours=hours_lag)).rename(
                    {"target": f"target_{hours_lag}h"}
                ),
                on=[
                    "county",
                    "is_business",
                    "product_type",
                    "is_consumption",
                    "datetime",
                ],
                how="left",
            )

        for hours_lag in [2 * 24, 3 * 24, 7 * 24, 14 * 24]:
            df_features = df_features.join(
                df_target_all_type_sum.with_columns(pl.col("datetime") + pl.duration(hours=hours_lag)).rename(
                    {"target": f"target_all_type_sum_{hours_lag}h"}
                ),
                on=["county", "is_business", "is_consumption", "datetime"],
                how="left",
            )

            df_features = df_features.join(
                df_target_all_county_type_sum.with_columns(pl.col("datetime") + pl.duration(hours=hours_lag)).rename(
                    {"target": f"target_all_county_type_sum_{hours_lag}h"}
                ),
                on=["is_business", "is_consumption", "datetime"],
                how="left",
                suffix=f"_all_county_type_sum_{hours_lag}h",
            )

        cols_for_stats = [f"target_{hours_lag}h" for hours_lag in hours_list[:4]]

        df_features = df_features.with_columns(
            df_features.select(cols_for_stats).mean(axis=1).alias("target_mean"),
            df_features.select(cols_for_stats).transpose().std().transpose().to_series().alias("target_std"),
        )

        for target_prefix, lag_nominator, lag_denomonator in [
            ("target", 24 * 7, 24 * 14),
            ("target", 24 * 2, 24 * 9),
            ("target", 24 * 3, 24 * 10),
            ("target", 24 * 2, 24 * 3),
            ("target_all_type_sum", 24 * 2, 24 * 3),
            ("target_all_type_sum", 24 * 7, 24 * 14),
            ("target_all_county_type_sum", 24 * 2, 24 * 3),
            ("target_all_county_type_sum", 24 * 7, 24 * 14),
        ]:
            df_features = df_features.with_columns(
                (
                    pl.col(f"{target_prefix}_{lag_nominator}h") / (pl.col(f"{target_prefix}_{lag_denomonator}h") + 1e-3)
                ).alias(f"{target_prefix}_ratio_{lag_nominator}_{lag_denomonator}")
            )

        return df_features

    def _reduce_memory_usage(self, df_features):
        df_features = df_features.with_columns(pl.col(pl.Float64).cast(pl.Float32))
        return df_features

    def _drop_columns(self, df_features):
        df_features = df_features.drop("datetime", "hour", "dayofyear")
        return df_features

    def _to_pandas(self, df_features, y):
        cat_cols = [
            "county",
            "is_business",
            "product_type",
            "is_consumption",
            "segment",
        ]

        if y is not None:
            df_features = pd.concat([df_features.to_pandas(), y.to_pandas()], axis=1)
        else:
            df_features = df_features.to_pandas()

        df_features = df_features.set_index("row_id")
        df_features[cat_cols] = df_features[cat_cols].astype("category")

        return df_features

    # added some new features here
    def _additional_features(self, df):
        for col in [
            "temperature",
            "dewpoint",
            "10_metre_u_wind_component",
            "10_metre_v_wind_component",
        ]:
            for window in [1]:
                df[f"{col}_diff_{window}"] = df.groupby(["county", "is_consumption", "product_type", "is_business"])[
                    col
                ].diff(window)
        return df

    def _log_outliers(self, df):
        l1 = ["installed_capacity", "target_mean", "target_std"]
        for i in l1:
            df = df.with_columns([(f"log_{i}", pl.when(df[i] != 0).then(np.log(pl.col(i))).otherwise(0))])
        return df

    def generate_features(self, df_prediction_items, isTrain):
        if "target" in df_prediction_items.columns:
            df_prediction_items, y = (
                df_prediction_items.drop("target"),
                df_prediction_items.select("target"),
            )
        else:
            y = None

        df_features = df_prediction_items.with_columns(
            pl.col("datetime").cast(pl.Date).alias("date"),
        )

        for add_features in [
            self._general_features,
            self._client_features,
            self._forecast_weather_features,
            self._historical_weather_features,
            self._target_features,
            self._holidays_features,
            self._log_outliers,
            self._reduce_memory_usage,
            self._drop_columns,
        ]:
            df_features = add_features(df_features)

        df_features = self._to_pandas(df_features, y)
        df_features = self._additional_features(df_features)

        return df_features
