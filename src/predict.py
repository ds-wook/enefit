from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from omegaconf import DictConfig

import enefit
from data import Warehouse
from features import FeatureEngineer

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
        store = Warehouse()
        feat_gen = FeatureEngineer(data=store)
        store.update_data(
            df_client_new=df_new_client,
            df_gas_price_new=df_new_gas_prices,
            df_elec_price_new=df_new_electricity_prices,
            df_forecast_new=df_new_forecast_weather,
            df_hist_weather_new=df_new_historical_weather,
            df_target_new=df_new_target,
        )
        df_test = store.preprocess_test(df_test)

        df_test_feats = feat_gen.generate_features(df_test, False)

        df_test_feats.drop(columns=["date"], inplace=True)

        if "literal" in df_test_feats.columns:
            df_test_feats.drop(columns=["literal"], inplace=True)

        m1 = joblib.load(Path(cfg.models.path) / "model_consumption.pkl")
        m2 = joblib.load(Path(cfg.models.path) / "model_production.pkl")

        df_sample_prediction["target"] = predict_model(df_test_feats, 48, m1, m2)

        env.predict(df_sample_prediction)


if __name__ == "__main__":
    _main()
