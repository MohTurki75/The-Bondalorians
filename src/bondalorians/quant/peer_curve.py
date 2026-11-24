"""Courbe des pairs : régression ridge du log du Z-Spread sur les caractéristiques.

Modèle volontairement lisible. Il sert de référence plancher, de support aux
ajustements des comparables et de base à la cascade en points de base.

Version de base : notation (et son carré), maturité, interaction notation ×
maturité, option de remboursement, séniorité, secteur, devise, type de coupon.

Version enrichie (`enriched=True`), chaque terme restant explicable :
- courbure de maturité (log de la maturité au carré) : les courbes s'aplatissent ;
- sensibilité à la notation propre à chaque secteur ;
- prime de séniorité qui varie avec la notation ;
- pays et taille de la souche (liquidité).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICALS = ["seniority", "sector", "currency", "coupon_type"]
NOTCH_CENTER = 9.0  # BBB : les interactions sont centrées sur le milieu de l'échelle
AMOUNT_CENTER = 20.0  # log de l'encours (environ 500 millions)

# Regroupement des colonnes de la régression pour la cascade explicative.
GROUP_LABELS = {
    "notch": "Notation",
    "log_t": "Maturité",
    "seniority": "Séniorité",
    "sector": "Secteur",
    "currency": "Devise",
    "country": "Pays",
    "coupon_type": "Type de coupon",
    "callable": "Option de remboursement anticipé",
    "size": "Taille de la souche",
    "esg": "ESG",
    "age": "Ancienneté de l'émission",
    "duration": "Duration",
    "vert": "Obligation verte",
    "holdco": "Dette de holding",
    "call": "Call avant l'échéance",
}
# Variables supplémentaires possibles (jour 1), activées par `extra` : colonne -> transformation.
EXTRA_FEATURES = {"esg": "esg_score", "age": "age_years", "duration": "duration", "vert": "green_flag",
                  "holdco": "holdco_flag", "call": "call_ratio"}


class PeerCurve:
    def __init__(self, ridge: float = 0.01, enriched: bool = False, extra: tuple = ()):
        self.ridge = ridge
        self.enriched = enriched
        self.extra = tuple(extra)

    def _blocks(self, df: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
        notch = df["notch"].to_numpy(dtype=float)
        log_t = df["log_t"].to_numpy(dtype=float)
        blocks = [
            ("_const", np.ones(len(df))), ("notch", notch), ("notch", notch**2),
            ("log_t", log_t), ("log_t", log_t * notch), ("callable", df["callable"].to_numpy(dtype=float)),
        ]
        for column in CATEGORICALS:
            values = df[column].to_numpy()
            blocks += [(column, (values == level).astype(float)) for level in self.levels_[column]]
        if self.enriched:
            centered = notch - NOTCH_CENTER
            blocks.append(("log_t", log_t**2))
            sectors = df["sector"].to_numpy()
            blocks += [("sector", (sectors == level) * centered) for level in self.levels_["sector"]]
            blocks.append(("seniority", df["sen_rank"].to_numpy(dtype=float) * centered))
            countries = df["country"].to_numpy()
            blocks += [("country", (countries == level).astype(float)) for level in self.levels_["country"]]
            blocks.append(("size", df["log_amt"].to_numpy(dtype=float) - AMOUNT_CENTER))
        for name in self.extra:
            values = self._extra_values(df, name)
            if values is not None:
                blocks.append((name, values))
        return blocks

    def _extra_values(self, df: pd.DataFrame, name: str) -> np.ndarray | None:
        """Variable supplémentaire centrée ; une valeur absente prend la moyenne (effet nul)."""
        column = EXTRA_FEATURES[name]
        if not hasattr(self, "extra_means_") or name not in self.extra_means_:
            return None
        if column not in df.columns:
            return np.zeros(len(df))
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if name == "age":
            values = np.log1p(np.clip(values, 0, None))
        return np.nan_to_num(values - self.extra_means_[name], nan=0.0)

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([column for _, column in self._blocks(df)])

    def fit(self, df: pd.DataFrame) -> "PeerCurve":
        columns = CATEGORICALS + (["country"] if self.enriched else [])
        self.levels_ = {c: sorted(df[c].unique()) for c in columns}
        self.extra_means_ = {}
        for name in self.extra:
            column = EXTRA_FEATURES[name]
            if column in df.columns and pd.to_numeric(df[column], errors="coerce").notna().mean() > 0.5:
                values = pd.to_numeric(df[column], errors="coerce")
                values = np.log1p(values.clip(lower=0)) if name == "age" else values
                self.extra_means_[name] = float(values.mean())
        blocks = self._blocks(df)
        self.groups_ = [group for group, _ in blocks]
        x = np.column_stack([column for _, column in blocks])
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
            if group not in self.group_means_:
                continue
            total = sum(x[i] for i, g in enumerate(self.groups_) if g == group)
            out[group] = float(total - self.group_means_[group])
        return out
