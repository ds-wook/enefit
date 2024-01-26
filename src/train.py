from __future__ import annotations

import warnings
from pathlib import Path

import hydra
import joblib
import lightgbm as lgb
from omegaconf import DictConfig
from sklearn.ensemble import VotingRegressor

from data import DataStorage
from features import FeatureEngineer
from modeling import fit_model


@hydra.main(config_path="../config/", config_name="train")
def _main(cfg: DictConfig):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        data_storage = DataStorage(cfg)
        feat_gen = FeatureEngineer(data_storage=data_storage)
        df_train = feat_gen.generate_features(data_storage.df_data)
        df_train = df_train[df_train["target"].notnull()]

        # Train model
        model_consumption = VotingRegressor(
            [
                (
                    f"clgb_{i}",
                    lgb.LGBMRegressor(**cfg.models.params, random_state=i),
                )
                for i in range(12)
            ],
            verbose=True,
        )

        model_production = VotingRegressor(
            [
                (
                    f"plgb_{i}",
                    lgb.LGBMRegressor(**cfg.models.params, random_state=i),
                )
                for i in range(12)
            ],
            verbose=True,
        )

        model_consumption, model_production = fit_model(df_train, model_consumption, model_production)

        joblib.dump(model_consumption, Path(cfg.models.path) / f"{cfg.models.model_consumption_diff}")
        joblib.dump(model_production, Path(cfg.models.path) / f"{cfg.models.model_production_diff}")


if __name__ == "__main__":
    _main()
