"""Modèle challenger : CatBoost sur le log du Z-Spread, avec contraintes de monotonie.

L'émetteur n'est volontairement pas une variable : le modèle doit généraliser à
un émetteur jamais vu. CatBoost est une dépendance optionnelle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC = ["notch", "log_t", "sen_rank", "log_amt", "callable_num"]
CATEGORICAL = ["sector", "country", "currency", "coupon_type", "seniority"]


def catboost_available() -> bool:
    try:
        import catboost  # noqa: F401
    except ImportError:
        return False
    return True


def _features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame({
        "notch": df["notch"].astype(float),
        "log_t": df["log_t"].astype(float),
        "sen_rank": df["sen_rank"].astype(float),
        "log_amt": df["log_amt"].astype(float),
        "callable_num": df["callable"].astype(float),
    })
    for column in CATEGORICAL:
        x[column] = df[column].astype(str).to_numpy()
    return x


class SpreadGBM:
    def __init__(self, iterations: int = 300, depth: int = 5, learning_rate: float = 0.08, seed: int = 7):
        self.params = dict(iterations=iterations, depth=depth, learning_rate=learning_rate, random_seed=seed)

    def fit(self, df: pd.DataFrame) -> "SpreadGBM":
        from catboost import CatBoostRegressor

        self.model_ = CatBoostRegressor(
            **self.params,
            loss_function="RMSE",
            cat_features=CATEGORICAL,
            # Le spread ne baisse pas quand la notation ou la séniorité se dégrade.
            monotone_constraints={"notch": 1, "sen_rank": 1},
            verbose=False,
            allow_writing_files=False,
        )
        self.model_.fit(
            _features(df),
            np.log(df["z_spread"].to_numpy(dtype=float)),
            sample_weight=df["obs_weight"].to_numpy(dtype=float),
        )
        return self

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(_features(df)), dtype=float)

    def shap_log(self, target: dict) -> dict[str, float]:
        """Contributions SHAP en log (donc multiplicatives sur le spread)."""
        from catboost import Pool

        x = _features(pd.DataFrame([target]))
        values = self.model_.get_feature_importance(Pool(x, cat_features=CATEGORICAL), type="ShapValues")[0]
        return {name: float(v) for name, v in zip(list(x.columns), values[:-1])}
