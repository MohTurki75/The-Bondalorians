"""Courbe des pairs : régression ridge du log du Z-Spread sur les caractéristiques.

Modèle volontairement simple et lisible. Il sert de référence plancher, de
support aux ajustements des comparables et de base à la cascade en points de base.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICALS = ["seniority", "sector", "currency", "coupon_type"]

# Regroupement des colonnes de la régression pour la cascade explicative.
GROUP_LABELS = {
    "notch": "Notation",
    "log_t": "Maturité",
    "seniority": "Séniorité",
    "sector": "Secteur",
    "currency": "Devise",
    "coupon_type": "Type de coupon",
    "callable": "Option de remboursement anticipé",
}


class PeerCurve:
    def __init__(self, ridge: float = 0.01):
        self.ridge = ridge

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        notch = df["notch"].to_numpy(dtype=float)
        log_t = df["log_t"].to_numpy(dtype=float)
        blocks = [
            np.ones(len(df)), notch, notch**2, log_t, log_t * notch,
            df["callable"].to_numpy(dtype=float),
        ]
        for column in CATEGORICALS:
            values = df[column].to_numpy()
            blocks.extend((values == level).astype(float) for level in self.levels_[column])
        return np.column_stack(blocks)

    def fit(self, df: pd.DataFrame) -> "PeerCurve":
        self.levels_ = {c: sorted(df[c].unique()) for c in CATEGORICALS}
        self.groups_ = ["_const", "notch", "notch", "log_t", "log_t", "callable"]
        for column in CATEGORICALS:
            self.groups_.extend([column] * len(self.levels_[column]))
        x = self._design(df)
        y = np.log(df["z_spread"].to_numpy(dtype=float))
        weights = df["obs_weight"].to_numpy(dtype=float)
        penalty = self.ridge * np.eye(x.shape[1])
        penalty[0, 0] = 0.0  # la constante n'est pas pénalisée
        xw = x * weights[:, None]
        self.coef_ = np.linalg.solve(x.T @ xw + penalty, xw.T @ y)
        # Contributions moyennes par groupe : définissent « l'obligation moyenne ».
        contributions = x * self.coef_
        self.group_means_ = {
            g: float(contributions[:, [i for i, gg in enumerate(self.groups_) if gg == g]].sum(1).mean())
            for g in set(self.groups_)
        }
        self.mean_log_ = float((x @ self.coef_).mean())
        return self

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        return self._design(df) @ self.coef_

    def group_contributions(self, target: dict) -> dict[str, float]:
        """Écart en log, par groupe, entre la cible et l'obligation moyenne de l'univers."""
        x = self._design(pd.DataFrame([target]))[0] * self.coef_
        out = {}
        for group in GROUP_LABELS:
            total = sum(x[i] for i, g in enumerate(self.groups_) if g == group)
            out[group] = float(total - self.group_means_[group])
        return out
