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
        feat_gen = FeatureEngineer(data=data_storage)
        df_train = feat_gen.generate_features(data_storage.df_data, True)
        df_train = df_train[df_train["target"].notnull()]

        # dropping column
        df_train = df_train.drop(columns=["date"])
        df_train = df_train.drop(columns=["literal"]) if "literal" in df_train.columns else df_train

        # Train model
        model_consumption = VotingRegressor(
            [
                (
                    f"clgb_{i}",
                    lgb.LGBMRegressor(**cfg.models.params, random_state=i),
                )
                for i in range(15)
            ],
            verbose=True,
        )

        model_production = VotingRegressor(
            [
                (
                    f"plgb_{i}",
                    lgb.LGBMRegressor(**cfg.models.params, random_state=i),
                )
                for i in range(15)
            ],
            verbose=True,
        )

        m1, m2 = fit_model(df_train, 48, model_consumption, model_production)

        joblib.dump(m1, Path(cfg.models.path) / f"{cfg.models.model_consumption}")
        joblib.dump(m2, Path(cfg.models.path) / f"{cfg.models.model_production}")


if __name__ == "__main__":
    _main()
