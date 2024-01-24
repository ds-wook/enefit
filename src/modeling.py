import gc

import pandas as pd
from sklearn.ensemble import VotingRegressor


def fit_model(
    train_feats: pd.DataFrame, model_consumption: VotingRegressor, model_production: VotingRegressor
) -> tuple[VotingRegressor]:
    mask = train_feats["is_consumption"] == 1
    model_consumption.fit(
        X=train_feats[mask].drop(columns=["target"]),
        y=train_feats[mask]["target"],
    )
    gc.collect()

    mask = train_feats["is_consumption"] == 0
    model_production.fit(
        X=train_feats[mask].drop(columns=["target"]),
        y=train_feats[mask]["target"],
    )
    gc.collect()

    return model_consumption, model_production
