"""Méthode de l'entonnoir : panier d'obligations par variables les plus explicatives.

Démarche :

1. **Courbe parfaite** : si l'émetteur a au moins 2 autres obligations de même
   notation, séniorité et devise, on interpole linéairement en maturité entre
   elles (après correction des petites différences restantes).
2. **Sinon, entonnoir** : les variables sont classées de la plus explicative à
   la moins explicative du spread (part de la dispersion des log-spreads
   expliquée par chaque variable seule, mesurée sur l'univers). En partant de
   tout l'univers, on garde à chaque étape les obligations qui partagent la
   valeur de la cible sur cette variable, tant qu'il en reste au moins
   `min_basket`. Une variable qui viderait le panier n'est pas imposée : elle
   devient une variable « manquante ».
3. **Correction** : chaque obligation du panier est corrigée de ce qui la
   distingue de la cible (variables manquantes, et écarts fins à l'intérieur
   des catégories imposées) par l'impact de chaque variable sur le spread,
   donné par les coefficients de la courbe des pairs.
4. **Barycentre** : l'estimation est la moyenne des log-spreads corrigés du
   panier.

Deux variantes : maturité exclue du classement (elle est seulement corrigée)
ou incluse (elle peut aussi restreindre le panier, par tranche de maturité).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from bondalorians.quant.peer_curve import GROUP_LABELS

MATURITY_BINS = [0, 3, 5, 7, 10, np.inf]
MATURITY_LABELS = ["moins de 3 ans", "3 à 5 ans", "5 à 7 ans", "7 à 10 ans", "plus de 10 ans"]
VARIABLE_LABELS = {
    "notation": "Notation (lettre)", "emetteur": "Émetteur", "seniorite": "Séniorité", "secteur": "Secteur",
    "devise": "Devise", "pays": "Pays", "coupon": "Type de coupon", "remboursable": "Remboursable",
    "maturite": "Maturité (tranche)", "taille": "Taille (tiers)",
}
DISPLAY_POINTS = 12


def _letter(notch) -> np.ndarray:
    notch = np.asarray(notch, dtype=float)
    bins = [0, 1, 4, 7, 10, 13, 16, np.inf]
    labels = np.array(["AAA", "AA", "A", "BBB", "BB", "B", "CCC"])
    return labels[np.clip(np.digitize(notch, bins[1:-1], right=True), 0, len(labels) - 1)]


def _keys(frame: pd.DataFrame, size_cuts: np.ndarray) -> dict[str, np.ndarray]:
    """Valeur de chaque variable de l'entonnoir, pour un ensemble d'obligations."""
    maturity = np.digitize(frame["maturity_years"].to_numpy(dtype=float), MATURITY_BINS[1:-1], right=False)
    return {
        "notation": _letter(frame["notch"]),
        "emetteur": frame["issuer"].astype(str).to_numpy(),
        "seniorite": frame["seniority"].astype(str).to_numpy(),
        "secteur": frame["sector"].astype(str).to_numpy(),
        "devise": frame["currency"].astype(str).to_numpy(),
        "pays": frame["country"].astype(str).to_numpy(),
        "coupon": frame["coupon_type"].astype(str).to_numpy(),
        "remboursable": frame["callable"].astype(bool).to_numpy(),
        "maturite": maturity,
        "taille": np.digitize(frame["log_amt"].to_numpy(dtype=float), size_cuts),
    }


def variable_ranking(estimator, include_maturity: bool) -> pd.DataFrame:
    """Part de la dispersion des log-spreads expliquée par chaque variable seule."""
    cache = getattr(estimator, "_funnel_cache", {})
    key = (estimator.model_version_, include_maturity)
    if key in cache:
        return cache[key]
    df = estimator.df_
    size_cuts = np.quantile(df["log_amt"], [1 / 3, 2 / 3])
    keys = _keys(df, size_cuts)
    y = np.log(df["z_spread"].to_numpy(dtype=float))
    total = ((y - y.mean()) ** 2).sum()
    rows = []
    for name, values in keys.items():
        if name == "maturite" and not include_maturity:
            continue
        means = pd.Series(y).groupby(values).transform("mean").to_numpy()
        rows.append({"variable": name, "libelle": VARIABLE_LABELS[name],
                     "part_expliquee": float(1 - ((y - means) ** 2).sum() / total)})
    ranking = pd.DataFrame(rows).sort_values("part_expliquee", ascending=False).reset_index(drop=True)
    ranking.attrs["size_cuts"] = size_cuts
    cache[key] = ranking
    estimator._funnel_cache = cache
    return ranking


def _group_corrections(peer, targets: pd.DataFrame, points: pd.DataFrame) -> dict[str, np.ndarray]:
    """Correction par groupe de variables (en log), point par point."""
    diff = (peer._design(targets) - peer._design(points)) * peer.coef_
    groups = np.array(peer.groups_)
    return {g: diff[:, groups == g].sum(axis=1) for g in GROUP_LABELS if g in set(peer.groups_)}


def funnel(estimator, t: dict, exclude_isin=None, include_maturity: bool = False,
           min_basket: int = 5, explain: bool = True) -> dict:
    a, df, peer = estimator.a_, estimator.df_, estimator.peer_
    keep = np.ones(len(df), dtype=bool) if exclude_isin is None else a["isin"] != exclude_isin
    target_frame = pd.DataFrame([t])

    # 1. Courbe parfaite de l'émetteur.
    perfect = keep & (a["issuer"] == t["issuer"]) & (a["notch"] == t["notch"]) \
        & (a["seniority"] == t["seniority"]) & (a["currency"] == t["currency"])
    if perfect.sum() >= 2:
        from bondalorians.quant.topdown import _interpolate

        idx = np.flatnonzero(perfect)
        at_point = pd.DataFrame([{**t, "log_t": lt} for lt in a["log_t"][idx]])
        adjusted = np.log(a["z_spread"][idx]) + peer.predict_log(at_point) - peer.predict_log(df.iloc[idx])

        def curve_at(maturity: float) -> float:
            return float(peer.predict_log(pd.DataFrame([{**t, "log_t": math.log(maturity)}]))[0])

        value, used, method = _interpolate(a["maturity_years"][idx], adjusted, float(t["maturity_years"]),
                                           curve_at, smooth=False)
        rows = idx[used]
        return _output(estimator, t, value, rows, adjusted[used], f"courbe parfaite de l'émetteur ({method})",
                       [{"variable": "Émetteur, notation, séniorité et devise", "imposee": True,
                         "restantes": int(len(idx))}], explain, perfect_curve=True)

    # 2. Entonnoir.
    ranking = variable_ranking(estimator, include_maturity)
    keys = _keys(df, ranking.attrs["size_cuts"])
    target_keys = {k: v[0] for k, v in _keys(target_frame, ranking.attrs["size_cuts"]).items()}
    basket = keep.copy()
    steps = []
    for row in ranking.itertuples():
        candidate = basket & (keys[row.variable] == target_keys[row.variable])
        applied = int(candidate.sum()) >= min_basket
        if applied:
            basket = candidate
        steps.append({"variable": row.libelle, "part_expliquee": round(row.part_expliquee, 3),
                      "imposee": applied, "restantes": int(basket.sum())})
    idx = np.flatnonzero(basket)
    if len(idx) == 0:
        value = float(peer.predict_log(target_frame)[0])
        return {"z_spread_bp": round(math.exp(value), 1), "log": value, "method": "courbe des pairs seule",
                "steps": steps, "points": [], "corrections_pct": {}, "basket_size": 0, "dispersion_pct": None,
                "perfect_curve": False}

    # 3 et 4. Correction de chaque point jusqu'à la cible, puis barycentre.
    targets = pd.concat([target_frame] * len(idx), ignore_index=True)
    corrected = np.log(a["z_spread"][idx]) + peer.predict_log(targets) - peer.predict_log(df.iloc[idx])
    value = float(np.mean(corrected))
    return _output(estimator, t, value, idx, corrected, f"barycentre de {len(idx)} obligations corrigées",
                   steps, explain)


def _output(estimator, t, value, rows, corrected, method, steps, explain, perfect_curve=False) -> dict:
    a, df, peer = estimator.a_, estimator.df_, estimator.peer_
    result = {"z_spread_bp": round(math.exp(value), 1), "log": value, "method": method, "steps": steps,
              "basket_size": int(len(rows)), "perfect_curve": perfect_curve,
              "dispersion_pct": round(100 * float(np.std(corrected)), 1) if len(rows) > 1 else None}
    if not explain:
        return result
    targets = pd.concat([pd.DataFrame([t])] * len(rows), ignore_index=True)
    corrections = _group_corrections(peer, targets, df.iloc[rows])
    result["corrections_pct"] = {
        GROUP_LABELS[g]: round(100 * float(np.mean(v)), 1) for g, v in corrections.items()
        if abs(float(np.mean(v))) >= 0.005
    }
    order = np.argsort(np.abs(a["log_t"][rows] - t["log_t"]))[:DISPLAY_POINTS]
    result["points"] = [{
        "isin": a["isin"][r], "issuer": a["issuer"][r], "rating": df["rating"].iat[r],
        "seniority": a["seniority"][r], "sector": a["sector"][r], "currency": a["currency"][r],
        "maturity_years": round(float(a["maturity_years"][r]), 2),
        "z_spread_bp": round(float(a["z_spread"][r]), 1),
        "corrected_z_spread_bp": round(math.exp(corrected[j]), 1),
    } for j, r in zip(order, np.asarray(rows)[order])]
    return result
