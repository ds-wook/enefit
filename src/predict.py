from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import VotingRegressor

from data import DataStorage
from features import FeatureEngineer

try:
    import enefit

except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install enefit package from")


def predict_model(
    df_features: pd.DataFrame, hours_lag: int, model_consumption: VotingRegressor, model_production: VotingRegressor
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
    env = enefit.make_env()
    iter_test = env.iter_test()
    model_consumption = joblib.load(Path(cfg.models.path) / f"{cfg.models.model_consumption}")
    model_production = joblib.load(Path(cfg.models.path) / f"{cfg.models.model_production}")

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
        data_storage = DataStorage(cfg)
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
        df_test_feats = feat_gen.generate_features(df_test, False)
        df_test_feats.drop(columns=["date", "literal"], inplace=True)

        preds = predict_model(df_test_feats, 48, model_consumption, model_production)

        df_sample_prediction["target"] = preds

        env.predict(df_sample_prediction)


if __name__ == "__main__":
    _main()
