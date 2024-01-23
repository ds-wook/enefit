from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from omegaconf import DictConfig

from data import DataStorage
from features import FeatureEngineer

try:
    import enefit

except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install enefit package from")

env = enefit.make_env()
iter_test = env.iter_test()
Model = list[LGBMRegressor | CatBoostRegressor]


def predict_model(
    df_features: pd.DataFrame, hours_lag: int, model_consumption: Model, model_production: Model
) -> np.ndarray:
    predictions = np.zeros(len(df_features))

    mask = df_features["is_consumption"] == 1
    predictions[mask.values] = np.clip(
        df_features[mask][f"target_{hours_lag}h"].fillna(0).values + model_consumption.predict(df_features[mask]),
        0,
        np.inf,
    )

    mask = df_features["is_consumption"] == 0
    predictions[mask.values] = np.clip(
        df_features[mask][f"target_{hours_lag}h"].fillna(0).values + model_production.predict(df_features[mask]),
        0,
        np.inf,
    )

    return predictions


@hydra.main(config_path="../config/", config_name="predict")
def _main(cfg: DictConfig):
    for (
        df_test,
        df_new_target,
        df_new_client,
        df_new_historical_weather,
        df_new_forecast_weather,
        df_new_electricity_prices,
        df_new_gas_prices,
        df_sample_prediction,
    ) in iter_test:
        data_storage = DataStorage()
        feat_gen = FeatureEngineer(data=data_storage)

        data_storage.update_with_new_data(
            df_new_client=df_new_client,
            df_new_gas_prices=df_new_gas_prices,
            df_new_electricity_prices=df_new_electricity_prices,
            df_new_forecast_weather=df_new_forecast_weather,
            df_new_historical_weather=df_new_historical_weather,
            df_new_target=df_new_target,
        )

        # separately generate test features for both models
        df_test = data_storage.preprocess_test(df_test)

        # df_test_features = features_generator.generate_features(df_test)

        df_test_feats = feat_gen.generate_features(df_test, False)

        # print(set(df_test_features.columns) - set(df_test_feats.columns))
        # print(set(df_test_feats.columns) - set(df_test_features.columns))
        # print("------")

        df_test_feats.drop(columns=["date", "literal"], inplace=True)

        df_sample_prediction["target"] = 0

        env.predict(df_sample_prediction)


if __name__ == "__main__":
    _main()
