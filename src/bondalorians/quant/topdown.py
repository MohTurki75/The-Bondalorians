"""Lecture descendante (top-down) : la démarche d'un trader crédit.

On cherche le groupe d'obligations le plus proche de la cible qui a assez de
points, du plus précis au plus général :

 1. même émetteur, même devise et même séniorité (au moins 2 points) ;
 2. même émetteur, autre devise ou séniorité (au moins 1 point) ;
 3. même secteur, devise, lettre de notation et séniorité (au moins 3) ;
 4. même secteur, lettre et séniorité, toute devise (au moins 3) ;
 5. même secteur et séniorité, notation à ±3 crans (au moins 3) ;
 6. même séniorité et devise, notation à ±2 crans (au moins 3) ;
 7. sinon, la courbe des pairs seule.

Chaque point est corrigé, variable par variable, de ce qui le distingue de la
cible (hors maturité) avec les coefficients de la courbe des pairs. Le spread
est ensuite interpolé linéairement en maturité entre les points qui encadrent
la cible ; hors de leur plage, on part du point le plus proche et on applique
l'effet de maturité de la courbe.

Ce n'est pas le chiffre retenu : c'est une lecture de contrôle, plus parlante
pour un gérant. Mesurée hors échantillon, elle se trompe davantage que le
modèle A, parce qu'une interpolation entre deux points hérite de tout leur
bruit (voir `evaluate_top_down`).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from bondalorians.quant.peer_curve import GROUP_LABELS

MODES = ("strict", "lisse", "mixte")
ISSUER_LEVELS = (1, 2)
LEVEL_NAMES = {
    1: "même émetteur, même devise et séniorité",
    2: "même émetteur, autre devise ou séniorité",
    3: "même secteur, devise, lettre de notation et séniorité",
    4: "même secteur, lettre de notation et séniorité",
    5: "même secteur et séniorité, notation à ±3 crans",
    6: "même séniorité et devise, notation à ±2 crans",
    7: "courbe des pairs seule",
}
MAX_POINTS = 200


def _letter(notch: float) -> str:
    for limit, label in ((1, "AAA"), (4, "AA"), (7, "A"), (10, "BBB"), (13, "BB"), (16, "B")):
        if notch <= limit:
            return label
    return "CCC"


def _levels(a: dict, t: dict, keep: np.ndarray) -> list[tuple[int, np.ndarray, int]]:
    letters = np.array([_letter(n) for n in a["notch"]])
    same = {c: a[c] == t[c] for c in ("issuer", "currency", "seniority", "sector")}
    letter = letters == _letter(t["notch"])
    gap = np.abs(a["notch"] - t["notch"])
    return [
        (1, keep & same["issuer"] & same["currency"] & same["seniority"], 2),
        (2, keep & same["issuer"], 1),
        (3, keep & same["sector"] & same["currency"] & letter & same["seniority"], 3),
        (4, keep & same["sector"] & letter & same["seniority"], 3),
        (5, keep & same["sector"] & same["seniority"] & (gap <= 3), 3),
        (6, keep & same["seniority"] & same["currency"] & (gap <= 2), 3),
    ]


def top_down(estimator, t: dict, exclude_isin=None, mode: str = "lisse") -> dict:
    """Lecture descendante pour une cible préparée (dictionnaire au schéma interne).

    `mode` : "strict" interpole entre les deux points qui encadrent la cible ;
    "lisse" moyenne les deux points les plus proches de chaque côté ; "mixte"
    interpole strictement entre les obligations de l'émetteur et lisse ailleurs.
    """
    if mode not in MODES:
        raise ValueError(f"Mode inconnu : {mode} (attendu : {MODES}).")
    a, df, peer = estimator.a_, estimator.df_, estimator.peer_
    keep = np.ones(len(df), dtype=bool) if exclude_isin is None else a["isin"] != exclude_isin

    def curve_at(maturity: float) -> float:
        return float(peer.predict_log(pd.DataFrame([{**t, "log_t": math.log(maturity)}]))[0])

    for level, mask, need in _levels(a, t, keep):
        idx = np.flatnonzero(mask)
        if len(idx) < need:
            continue
        if len(idx) > MAX_POINTS:
            idx = idx[np.argsort(np.abs(a["log_t"][idx] - t["log_t"]))[:MAX_POINTS]]
        # Correction variable par variable : la cible placée à la maturité du point, moins le point.
        at_point = pd.DataFrame([{**t, "log_t": lt} for lt in a["log_t"][idx]])
        correction = peer.predict_log(at_point) - peer.predict_log(df.iloc[idx])
        adjusted = np.log(a["z_spread"][idx]) + correction
        x, x0 = a["maturity_years"][idx], float(t["maturity_years"])
        smooth = mode == "lisse" or (mode == "mixte" and level not in ISSUER_LEVELS)
        value, used, method = _interpolate(x, adjusted, x0, curve_at, smooth=smooth and len(idx) >= 3)
        return _result(estimator, t, level, value, idx, used, method, correction, adjusted, at_point)

    value = float(peer.predict_log(pd.DataFrame([t]))[0])
    return {"z_spread_bp": round(math.exp(value), 1), "log": value, "level": 7, "level_name": LEVEL_NAMES[7],
            "method": "courbe des pairs", "points": [], "corrections_pct": {}, "n_candidates": 0}


def _interpolate(x, y, x0, curve_at, smooth: bool):
    """Renvoie (valeur en log, positions des points utilisés, méthode)."""
    if smooth:
        left, right = np.flatnonzero(x <= x0), np.flatnonzero(x > x0)
        pick = np.concatenate([left[np.argsort(x0 - x[left])[:2]], right[np.argsort(x[right] - x0)[:2]]])
        values = [y[j] + curve_at(x0) - curve_at(x[j]) for j in pick]
        return float(np.mean(values)), pick, "moyenne des 2 points les plus proches de chaque côté"
    if x.min() <= x0 <= x.max() and x.max() > x.min():
        below = np.flatnonzero(x <= x0)
        above = np.flatnonzero(x >= x0)
        lo = below[np.argmax(x[below])]
        hi = above[np.argmin(x[above])]
        if x[hi] == x[lo]:
            return float(y[lo]), np.array([lo]), "point à la même maturité"
        weight = (x0 - x[lo]) / (x[hi] - x[lo])
        return float((1 - weight) * y[lo] + weight * y[hi]), np.array([lo, hi]), "interpolation linéaire"
    nearest = int(np.argmin(np.abs(x - x0)))
    return float(y[nearest] + curve_at(x0) - curve_at(x[nearest])), np.array([nearest]), \
        "extrapolation depuis le point le plus proche"


def _result(estimator, t, level, value, idx, used, method, correction, adjusted, at_point) -> dict:
    a, df, peer = estimator.a_, estimator.df_, estimator.peer_
    rows = idx[used]
    # Corrections moyennes par variable sur les points utilisés (en log, donc à peu près en %).
    by_group: dict[str, float] = {}
    for position, row in zip(used, rows):
        target_parts = peer.group_contributions(at_point.iloc[position].to_dict())
        point_parts = peer.group_contributions(df.iloc[row].to_dict())
        for group, label in GROUP_LABELS.items():
            if group in target_parts and group != "log_t":
                by_group[label] = by_group.get(label, 0.0) + target_parts[group] - point_parts[group]
    corrections = {k: round(100 * v / len(rows), 1) for k, v in by_group.items() if abs(v / len(rows)) >= 0.005}
    points = [{
        "isin": a["isin"][r], "issuer": a["issuer"][r], "rating": df["rating"].iat[r],
        "seniority": a["seniority"][r], "currency": a["currency"][r],
        "maturity_years": round(float(a["maturity_years"][r]), 2),
        "z_spread_bp": round(float(a["z_spread"][r]), 1),
        "corrected_z_spread_bp": round(math.exp(adjusted[p]), 1),
        "correction_pct": round(100 * (math.exp(correction[p]) - 1), 1),
    } for p, r in zip(used, rows)]
    return {"z_spread_bp": round(math.exp(value), 1), "log": value, "level": level,
            "level_name": LEVEL_NAMES[level], "method": method, "points": points,
            "corrections_pct": corrections, "n_candidates": int(len(idx))}


def evaluate_top_down(estimator, modes=MODES) -> dict[str, pd.DataFrame]:
    """Compare la lecture descendante au modèle A sur les plis exacts du backtest.

    L'estimateur doit être calibré. Renvoie l'erreur médiane (en %) par palier
    et par niveau de la lecture descendante.
    """
    if estimator.calibration_ is None:
        raise RuntimeError("Estimateur non calibré.")
    oof = estimator.calibration_["oof"][["row_id", "scheme", "key", "y_log", "log_a", "peer_log"]]
    folds = estimator.make_folds(np.random.default_rng(estimator.settings["calibration"]["seed"]))
    rows = []
    for scheme, fold_ids in folds.items():
        for fold in np.unique(fold_ids):
            test = np.flatnonzero(fold_ids == fold)
            sub = estimator.spawn(use_ml=False)
            sub.fit_base(estimator.df_[fold_ids != fold])
            for i in test:
                target = estimator.df_.iloc[i].to_dict()
                row = {"row_id": int(i), "scheme": scheme}
                for mode in modes:
                    reading = top_down(sub, target, mode=mode)
                    row[f"log_{mode}"] = reading["log"]
                    if mode == modes[0]:
                        row["niveau"] = f"{reading['level']}. {reading['level_name']}"
                rows.append(row)
    data = pd.DataFrame(rows).merge(oof, on=["row_id", "scheme"])
    columns = {"log_a": "Modèle A", "peer_log": "Courbe des pairs seule",
               **{f"log_{m}": f"Top-down {m}" for m in modes}}
    for column in columns:
        data["err_" + column] = (data[column] - data["y_log"]).abs()
    errors = [f"err_{c}" for c in columns]
    names = {f"err_{c}": label for c, label in columns.items()}
    by_tier = (100 * data.groupby("key")[errors].median()).rename(columns=names).round(1)
    by_tier.insert(0, "Cas", data.groupby("key").size())
    by_level = (100 * data.groupby("niveau")[errors].median()).rename(columns=names).round(1)
    by_level.insert(0, "Cas", data.groupby("niveau").size())
    return {"par_palier": by_tier.reset_index().rename(columns={"key": "Palier"}),
            "par_niveau": by_level.reset_index().rename(columns={"niveau": "Niveau du top-down"})}
