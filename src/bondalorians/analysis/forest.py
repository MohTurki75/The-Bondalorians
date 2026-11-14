"""Forêt aléatoire : modèle de comparaison, évalué exactement comme les autres.

Elle n'entre pas dans l'estimation. Elle sert à répondre à la question
« un autre modèle ML ferait-il mieux ? », sur les mêmes plis que le backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC = ["notch", "log_t", "sen_rank", "log_amt", "callable"]
CATEGORICAL = ["sector", "currency", "coupon_type", "country", "seniority"]
LABELS = {
    "notch": "Notation", "log_t": "Maturité", "sen_rank": "Rang de séniorité", "log_amt": "Taille",
    "callable": "Remboursable par anticipation", "sector": "Secteur", "currency": "Devise",
    "coupon_type": "Type de coupon", "country": "Pays", "seniority": "Séniorité (catégorie)",
}


class ForestEncoder:
    """Variables numériques telles quelles, catégories en indicatrices."""

    def fit(self, df: pd.DataFrame) -> "ForestEncoder":
        self.levels_ = {c: sorted(df[c].astype(str).unique()) for c in CATEGORICAL}
        self.columns_, self.groups_ = list(NUMERIC), list(NUMERIC)
        for column in CATEGORICAL:
            self.columns_ += [f"{column}={v}" for v in self.levels_[column]]
            self.groups_ += [column] * len(self.levels_[column])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks = [df[c].to_numpy(dtype=float)[:, None] for c in NUMERIC]
        for column in CATEGORICAL:
            values = df[column].astype(str).to_numpy()
            blocks.append(np.column_stack([(values == v).astype(float) for v in self.levels_[column]]))
        return np.hstack(blocks)


def make_forest(seed: int = 7):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=200, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=seed,
    )


def forest_out_of_fold(estimator) -> pd.DataFrame:
    """Prédictions hors échantillon de la forêt, sur les plis exacts du backtest."""
    df = estimator.df_
    seed = estimator.settings["calibration"]["seed"]
    folds = estimator.make_folds(np.random.default_rng(seed))  # mêmes plis que _calibrate
    y = np.log(df["z_spread"].to_numpy(dtype=float))
    weights = df["obs_weight"].to_numpy(dtype=float)
    encoder = ForestEncoder().fit(df)
    x = encoder.transform(df)
    rows = []
    for scheme, fold_ids in folds.items():
        for fold in np.unique(fold_ids):
            test = fold_ids == fold
            model = make_forest(seed).fit(x[~test], y[~test], sample_weight=weights[~test])
            for row_id, value in zip(np.flatnonzero(test), model.predict(x[test])):
                rows.append({"row_id": int(row_id), "scheme": scheme, "log_rf": float(value)})
    return pd.DataFrame(rows)


def grouped_permutation_importance(estimator, seed: int = 7) -> pd.DataFrame:
    """Hausse de l'erreur quand on mélange une variable (toutes ses indicatrices ensemble).

    Mesurée sur 20 % d'émetteurs jamais vus par la forêt.
    """
    df = estimator.df_
    rng = np.random.default_rng(seed)
    issuers = df["issuer"].unique()
    held = set(rng.choice(issuers, size=max(1, len(issuers) // 5), replace=False))
    test = df["issuer"].isin(held).to_numpy()
    encoder = ForestEncoder().fit(df)
    x, y = encoder.transform(df), np.log(df["z_spread"].to_numpy(dtype=float))
    model = make_forest(seed).fit(x[~test], y[~test])
    x_test, y_test = x[test], y[test]
    base = float(np.median(np.abs(model.predict(x_test) - y_test)))
    groups = np.array(encoder.groups_)
    rows = []
    for variable in NUMERIC + CATEGORICAL:
        columns = np.flatnonzero(groups == variable)
        increases = []
        for _ in range(3):
            shuffled = x_test.copy()
            shuffled[:, columns] = shuffled[rng.permutation(len(shuffled))][:, columns]
            increases.append(float(np.median(np.abs(model.predict(shuffled) - y_test))) - base)
        rows.append({"variable": LABELS[variable], "hausse_erreur_pct": round(100 * np.mean(increases), 2)})
    return pd.DataFrame(rows).sort_values("hausse_erreur_pct", ascending=False).reset_index(drop=True)


def fit_forest(estimator, seed: int = 7):
    """Forêt entraînée sur tout l'univers (pour les courbes d'effet)."""
    encoder = ForestEncoder().fit(estimator.df_)
    model = make_forest(seed).fit(
        encoder.transform(estimator.df_), np.log(estimator.df_["z_spread"].to_numpy(dtype=float)),
        sample_weight=estimator.df_["obs_weight"].to_numpy(dtype=float),
    )
    return encoder, model
