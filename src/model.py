import gc

import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

Model = list[LGBMRegressor | CatBoostRegressor]


def fit_model(
    train_feats: pd.DataFrame, hours_lag: int, model_consumption: Model, model_production: Model
) -> tuple[Model, Model]:
    mask = train_feats["is_consumption"] == 1
    model_consumption.fit(
        X=train_feats[mask].drop(columns=["target"]),
        y=train_feats[mask]["target"] - train_feats[mask][f"target_{hours_lag}h"].fillna(0),
    )
    gc.collect()

    mask = train_feats["is_consumption"] == 0
    model_production.fit(
        X=train_feats[mask].drop(columns=["target"]),
        y=train_feats[mask]["target"] - train_feats[mask][f"target_{hours_lag}h"].fillna(0),
    )
    gc.collect()

    return model_consumption, model_production
