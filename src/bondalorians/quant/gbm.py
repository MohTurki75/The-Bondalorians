"""Modèles CatBoost (dépendance optionnelle).

- SpreadGBM (modèle B) : prédit le log du Z-Spread, indépendamment du modèle A.
  Il sert de challenger et de contrôle de désaccord.
- SpreadCorrector (modèle C) : prédit l'erreur résiduelle du modèle A à partir
  des caractéristiques et de la qualité des comparables. Il corrige le modèle
  transparent au lieu de le concurrencer.

L'émetteur n'est jamais une variable : les modèles doivent généraliser à un
émetteur inconnu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL = ["sector", "country", "currency", "coupon_type", "seniority"]
CORE_FEATURES = ["log_a", "lam", "r_issuer", "r_peers", "n_eff", "mean_distance", "n_issuer", "filter_level"]


def catboost_available() -> bool:
    try:
        import catboost  # noqa: F401
    except ImportError:
        return False
    return True


def _features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame({
        "notch": df["notch"].to_numpy(dtype=float),
        "log_t": df["log_t"].to_numpy(dtype=float),
        "sen_rank": df["sen_rank"].to_numpy(dtype=float),
        "log_amt": df["log_amt"].to_numpy(dtype=float),
        "callable_num": df["callable"].to_numpy(dtype=float),
    })
    for column in CATEGORICAL:
        x[column] = df[column].astype(str).to_numpy()
    return x


def _corrector_features(df: pd.DataFrame, cores: list[dict]) -> pd.DataFrame:
    x = _features(df)
    for name in CORE_FEATURES:
        x[name] = np.array([core[name] for core in cores], dtype=float)
    return x


def _shap(model, x: pd.DataFrame) -> dict[str, float]:
    from catboost import Pool

    values = model.get_feature_importance(Pool(x, cat_features=CATEGORICAL), type="ShapValues")[0]
    return {name: float(v) for name, v in zip(list(x.columns), values[:-1])}


class SpreadGBM:
    def __init__(self, iterations: int = 300, depth: int = 5, learning_rate: float = 0.08, seed: int = 7,
                 monotone: bool = False):
        self.params = dict(iterations=iterations, depth=depth, learning_rate=learning_rate, random_seed=seed)
        self.monotone = monotone

    def fit(self, df: pd.DataFrame) -> "SpreadGBM":
        from catboost import CatBoostRegressor

        self.model_ = CatBoostRegressor(
            **self.params,
            loss_function="RMSE",
            cat_features=CATEGORICAL,
            # Contrainte « le spread ne baisse pas quand la notation ou la séniorité se dégrade ».
            # Désactivée par défaut : mesurée, elle crée un biais de −4 % (sous-estimation
            # systématique) et augmente l'erreur de 7 %. Le respect de la monotonie est vérifié
            # à part (tests et courbes d'effet de l'onglet Analyse).
            monotone_constraints={"notch": 1, "sen_rank": 1} if self.monotone else None,
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
        return _shap(self.model_, _features(pd.DataFrame([target])))


class SpreadCorrector:
    """Apprend `log(spread observé) - log(modèle A)` ; volontairement peu profond."""

    def __init__(self, iterations: int = 200, depth: int = 4, learning_rate: float = 0.05, seed: int = 7):
        self.params = dict(iterations=iterations, depth=depth, learning_rate=learning_rate, random_seed=seed)

    def fit(self, df: pd.DataFrame, cores: list[dict], residual_log: np.ndarray,
            weights: np.ndarray) -> "SpreadCorrector":
        from catboost import CatBoostRegressor

        self.model_ = CatBoostRegressor(
            **self.params, loss_function="RMSE", cat_features=CATEGORICAL,
            l2_leaf_reg=5.0, verbose=False, allow_writing_files=False,
        )
        self.model_.fit(_corrector_features(df, cores), residual_log, sample_weight=weights)
        return self

    def predict(self, df: pd.DataFrame, cores: list[dict]) -> np.ndarray:
        return np.asarray(self.model_.predict(_corrector_features(df, cores)), dtype=float)

    def shap_log(self, target: dict, core: dict) -> dict[str, float]:
        return _shap(self.model_, _corrector_features(pd.DataFrame([target]), [core]))
