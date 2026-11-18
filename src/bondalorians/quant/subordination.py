"""Dette subordonnée ancrée sur la dette senior du même émetteur.

Un trader valorise une dette subordonnée en deux étapes :

    spread subordonné = niveau de la dette senior de l'émetteur (ramené au profil de la cible)
                        + prime de subordination

1. **Niveau senior de l'émetteur.** Chaque obligation senior de l'émetteur est
   ramenée, par la courbe des pairs, à un profil commun « senior » (même
   notation, maturité, devise… que la cible, séniorité de référence). Leur écart
   moyen à la courbe donne le niveau senior propre à l'émetteur.
2. **Prime de subordination, mesurée à l'intérieur des émetteurs.** Pour chaque
   émetteur qui a à la fois de la dette senior et de la dette subordonnée, on
   mesure l'écart entre les deux, toutes autres choses égales par ailleurs. Ces
   écarts sont régressés sur la catégorie de séniorité et la notation. Mesurer
   la prime à l'intérieur d'un même émetteur évite de la confondre avec la
   qualité des émetteurs qui émettent de la dette subordonnée.

Le niveau senior est rétréci comme l'écart émetteur du modèle A
(λ = n / (n + k)), sauf si `shrink=False`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SENIOR_MAX_RANK = 1          # garanti (0) et senior (1)
REFERENCE_SENIORITY = "SENIOR_UNSECURED"
MIN_OBS_PER_CATEGORY = 15
RIDGE = 1.0


def is_subordinated(sen_rank: float) -> bool:
    return float(sen_rank) > SENIOR_MAX_RANK


def _as_senior(frame: pd.DataFrame) -> pd.DataFrame:
    """Le même profil, avec la séniorité de référence."""
    out = frame.copy()
    out["seniority"] = REFERENCE_SENIORITY
    out["sen_rank"] = 1.0
    return out


class SubordinationModel:
    def fit(self, estimator) -> "SubordinationModel":
        df, peer = estimator.df_, estimator.peer_
        log_z = np.log(df["z_spread"].to_numpy(dtype=float))
        # Écart à la courbe « comme si l'obligation était senior » : il contient
        # l'effet émetteur et, pour une dette subordonnée, la prime réelle.
        self.senior_resid_ = log_z - peer.predict_log(_as_senior(df))
        # Prime générique de la courbe des pairs (pour les catégories peu observées).
        self.generic_ = peer.predict_log(df) - peer.predict_log(_as_senior(df))

        rows = []
        frame = pd.DataFrame({"issuer": df["issuer"].to_numpy(), "currency": df["currency"].to_numpy(),
                              "rank": df["sen_rank"].to_numpy(), "seniority": df["seniority"].to_numpy(),
                              "notch": df["notch"].to_numpy(), "r": self.senior_resid_})
        for _, group in frame.groupby("issuer"):
            senior = group[group["rank"] <= SENIOR_MAX_RANK]
            junior = group[group["rank"] > SENIOR_MAX_RANK]
            if senior.empty or junior.empty:
                continue
            level = senior["r"].mean()
            for row in junior.itertuples():
                rows.append({"seniority": row.seniority, "notch": row.notch, "premium": row.r - level,
                             "n_senior": len(senior)})
        pairs = pd.DataFrame(rows)
        self.pairs_ = pairs
        self.categories_ = {}
        if pairs.empty:
            self.slope_ = 0.0
            return self
        counts = pairs["seniority"].value_counts()
        kept = [c for c in counts.index if counts[c] >= MIN_OBS_PER_CATEGORY]
        data = pairs[pairs["seniority"].isin(kept)]
        if data.empty:
            self.slope_ = 0.0
            return self
        # Prime = constante par catégorie + pente commune en notation (centrée sur BBB), ridge léger.
        x = np.column_stack([(data["seniority"] == c).astype(float) for c in kept]
                            + [data["notch"].to_numpy(dtype=float) - 9.0])
        penalty = RIDGE * np.eye(x.shape[1])
        penalty[:-1, :-1] *= 0.01
        coef = np.linalg.solve(x.T @ x + penalty, x.T @ data["premium"].to_numpy(dtype=float))
        self.categories_ = {c: float(coef[i]) for i, c in enumerate(kept)}
        self.slope_ = float(coef[-1])
        self.counts_ = {c: int(counts[c]) for c in kept}
        return self

    def premium(self, target: dict, generic: float) -> tuple[float, str]:
        """Prime de subordination en log, et sa source."""
        category = target["seniority"]
        if category in self.categories_:
            value = self.categories_[category] + self.slope_ * (float(target["notch"]) - 9.0)
            return value, f"mesurée sur {self.counts_[category]} obligations {category} d'émetteurs ayant aussi de la dette senior"
        return generic, "prime générique de la courbe des pairs (trop peu d'observations pour cette catégorie)"

    def anchor(self, estimator, target: dict, exclude_isin=None, shrink: bool = True) -> dict | None:
        """Estimation ancrée sur la dette senior de l'émetteur, ou None si impossible."""
        if not is_subordinated(target["sen_rank"]):
            return None
        a, peer = estimator.a_, estimator.peer_
        mask = (a["issuer"] == target["issuer"]) & (a["sen_rank"] <= SENIOR_MAX_RANK)
        if exclude_isin is not None:
            mask &= a["isin"] != exclude_isin
        same_currency = mask & (a["currency"] == target["currency"])
        if same_currency.any():
            mask = same_currency
        if not mask.any():
            return None
        n = int(mask.sum())
        level = float(np.mean(self.senior_resid_[mask]))
        weight = n / (n + estimator.k_) if shrink else 1.0
        frame = pd.DataFrame([target])
        senior_curve = float(peer.predict_log(_as_senior(frame))[0])
        generic = float(peer.predict_log(frame)[0]) - senior_curve
        premium, source = self.premium(target, generic)
        # Dette senior de l'émetteur ramenée à la maturité, la devise… de la cible, mais avec sa
        # propre notation (souvent meilleure que celle de la dette subordonnée).
        senior_profile = {**target, "notch": float(np.median(a["notch"][mask]))}
        senior_equivalent = float(peer.predict_log(_as_senior(pd.DataFrame([senior_profile])))[0]) + level
        return {
            "log": senior_curve + weight * level + premium,
            "senior_equivalent_log": senior_equivalent,
            "senior_bonds": n, "senior_level_log": level, "weight": weight,
            "premium_log": premium, "premium_source": source, "generic_premium_log": generic,
            "senior_isins": list(a["isin"][mask]),
        }
