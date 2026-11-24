"""Motifs dans les erreurs du système, pour orienter des agents.

Deux analyses, toutes deux sur les estimations hors échantillon du backtest :

1. **Poches de biais** : segments d'obligations (une ou deux caractéristiques :
   notation, séniorité, secteur, devise, pays, maturité, taille, ESG, ancienneté…)
   où l'erreur est systématiquement du même signe. Garde-fous contre les faux
   motifs :
   - au moins `min_bonds` obligations et `min_issuers` émetteurs dans le segment ;
   - test sur les moyennes par émetteur (les obligations d'un même émetteur ne
     sont pas indépendantes) ;
   - correction pour le grand nombre de segments testés (Benjamini-Hochberg),
     séparément pour les segments simples et les croisements ;
   - réplication : l'effet doit avoir le même signe et au moins la moitié de
     l'ampleur minimale dans chacune de deux moitiés d'émetteurs indépendantes.

2. **Table d'aiguillage** : pour chaque cas (notation investissement ou haut
   rendement × senior ou subordonnée × information sur l'émetteur), la méthode
   qui se trompe le moins, si l'écart est net et confirmé sur les deux moitiés
   d'émetteurs, et le facteur d'élargissement de l'intervalle si la couverture
   y est insuffisante.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from bondalorians.analysis.common import BUCKET_ORDER, rating_bucket

DIMENSIONS = {  # nom -> libellé ; calculées par `_segment_frame`
    "notation": "Notation", "seniorite": "Séniorité", "secteur": "Secteur", "devise": "Devise", "pays": "Pays",
    "maturite": "Maturité", "taille": "Taille", "remboursable": "Remboursable", "coupon": "Type de coupon",
    "esg": "ESG", "anciennete": "Ancienneté", "industrie": "Industrie", "structure": "Structure (holdco / opco)",
    "vert": "Obligation verte",
}
MATURITY_BINS = [0, 3, 5, 7, 10, 15, np.inf]
MATURITY_LABELS = ["< 3 ans", "3-5 ans", "5-7 ans", "7-10 ans", "10-15 ans", "> 15 ans"]
AGE_BINS = [-np.inf, 0.5, 2, 5, np.inf]
AGE_LABELS = ["< 6 mois", "6 mois-2 ans", "2-5 ans", "> 5 ans"]
METHODS = {"peer_log": "P · courbe des pairs", "log_a": "A · comparables", "log_b": "B · CatBoost",
           "log_c": "C · A corrigé", "log_s": "S · ancrage senior"}
UNSEEN = "emetteur_inconnu"


def segment_edges(df: pd.DataFrame) -> dict:
    """Seuils des tiers de taille et d'ESG, fixés sur l'univers (pour classer aussi une cible)."""
    edges = {"taille": [float(v) for v in np.quantile(df["log_amt"], [1 / 3, 2 / 3])]}
    if "esg_score" in df.columns and df["esg_score"].notna().mean() > 0.5:
        edges["esg"] = [float(v) for v in np.nanquantile(df["esg_score"], [1 / 3, 2 / 3])]
    if "age_years" in df.columns and df["age_years"].notna().mean() > 0.5:
        edges["anciennete"] = True
    return edges


def _tier(values, cuts, labels):
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    out = np.where(values <= cuts[0], labels[0], np.where(values <= cuts[1], labels[1], labels[2]))
    return pd.Series(np.where(values.isna(), "inconnu", out), index=values.index)


def _segment_frame(df: pd.DataFrame, edges: dict | None = None) -> pd.DataFrame:
    """Une colonne par dimension de segmentation, en libellés lisibles."""
    edges = edges or segment_edges(df)
    out = pd.DataFrame(index=df.index)
    out["notation"] = df["notch"].map(rating_bucket)
    out["seniorite"] = df["seniority"].astype(str)
    out["secteur"] = df["sector"].astype(str)
    out["devise"] = df["currency"].astype(str)
    out["pays"] = df["country"].astype(str)
    out["maturite"] = pd.cut(df["maturity_years"], MATURITY_BINS, labels=MATURITY_LABELS).astype(str)
    out["taille"] = _tier(df["log_amt"], edges["taille"], ["petite", "moyenne", "grande"]).to_numpy()
    out["remboursable"] = np.where(df["callable"].astype(bool), "oui", "non")
    out["coupon"] = df["coupon_type"].astype(str)
    if "esg" in edges and "esg_score" in df.columns:
        out["esg"] = _tier(df["esg_score"], edges["esg"], ["ESG faible", "ESG moyen", "ESG élevé"]).to_numpy()
    if "anciennete" in edges and "age_years" in df.columns:
        out["anciennete"] = pd.cut(df["age_years"], AGE_BINS, labels=AGE_LABELS).astype(str)
    for column, name in (("industry", "industrie"), ("structure", "structure")):
        if column in df.columns:
            out[name] = df[column].astype(str)
    if "green_flag" in df.columns:
        out["vert"] = np.where(df["green_flag"] == 1, "oui", "non")
    return out


def _benjamini_hochberg(pvalues: np.ndarray, rate: float) -> np.ndarray:
    n = len(pvalues)
    if n == 0:
        return np.array([], dtype=bool)
    order = np.argsort(pvalues)
    thresholds = rate * (np.arange(1, n + 1) / n)
    passed = pvalues[order] <= thresholds
    keep = np.zeros(n, dtype=bool)
    if passed.any():
        keep[order[: np.max(np.flatnonzero(passed)) + 1]] = True
    return keep


def _issuer_test(residual: np.ndarray, issuers: np.ndarray) -> tuple[float, float, int]:
    """Moyenne des écarts par émetteur, puis test de Student sur ces moyennes."""
    from scipy import stats

    means = pd.Series(residual).groupby(issuers).mean().to_numpy()
    if len(means) < 3 or np.std(means, ddof=1) == 0:
        return float(np.mean(means)), 1.0, len(means)
    t, p = stats.ttest_1samp(means, 0.0)
    return float(np.mean(means)), float(p), len(means)


def _cases(estimator, target: str) -> pd.DataFrame:
    """Cas du backtest avec leur écart (vrai − estimé, en log) et leurs segments."""
    cal = estimator.calibration_
    oof = cal["oof"].copy()
    if target == "retenu":
        column = {"A": "log_a", "B": "log_b", "C": "log_c", "P": "peer_log", "S": "log_s"}
        oof["estimate"] = [row[column[cal["champion"].get(row["key"], "A")]] for _, row in oof.iterrows()]
    else:
        oof["estimate"] = oof["peer_log"]
    oof["ecart"] = oof["y_log"] - oof["estimate"].astype(float)
    oof["situation"] = np.where(oof["key"] == UNSEEN, "émetteur inconnu", "émetteur connu")
    df = estimator.df_
    segments = _segment_frame(df).iloc[oof["row_id"].to_numpy()].reset_index(drop=True)
    oof = oof.reset_index(drop=True)
    oof["issuer"] = df["issuer"].to_numpy()[oof["row_id"].to_numpy()]
    return pd.concat([oof, segments], axis=1)


def find_pockets(estimator, target: str = "retenu", min_bonds: int = 30, min_issuers: int = 8,
                 min_effect: float = 0.03, fdr: float = 0.05, pairs: bool = True, seed: int = 7) -> pd.DataFrame:
    """Poches de biais confirmées. `target` : « retenu » (estimation finale) ou « courbe »."""
    cases = _cases(estimator, target)
    dims = [d for d in DIMENSIONS if d in cases.columns]
    combos = [(d,) for d in dims] + (list(itertools.combinations(dims, 2)) if pairs else [])
    rng = np.random.default_rng(seed)
    unique = cases["issuer"].unique()
    half_of = dict(zip(unique, rng.permutation(len(unique)) % 2))
    cases["moitie"] = cases["issuer"].map(half_of)

    rows = []
    for situation, data in cases.groupby("situation"):
        for combo in combos:
            for values, group in data.groupby(list(combo)):
                values = values if isinstance(values, tuple) else (values,)
                if len(group) < min_bonds or group["issuer"].nunique() < min_issuers:
                    continue
                discovery = group[group["moitie"] == 0]
                confirmation = group[group["moitie"] == 1]
                if discovery["issuer"].nunique() < 4 or confirmation["issuer"].nunique() < 4:
                    continue
                mean_d, p_d, _ = _issuer_test(discovery["ecart"].to_numpy(), discovery["issuer"].to_numpy())
                mean_c, p_c, _ = _issuer_test(confirmation["ecart"].to_numpy(), confirmation["issuer"].to_numpy())
                mean_all, p_all, n_iss = _issuer_test(group["ecart"].to_numpy(), group["issuer"].to_numpy())
                rows.append({
                    "situation": situation, "segment": " · ".join(f"{DIMENSIONS[d]} {v}" for d, v in zip(combo, values)),
                    "dimensions": dict(zip(combo, values)), "obligations": len(group), "emetteurs": n_iss,
                    "ecart_pct": 100 * (np.exp(mean_all) - 1), "ecart_decouverte_pct": 100 * (np.exp(mean_d) - 1),
                    "ecart_confirmation_pct": 100 * (np.exp(mean_c) - 1), "p_decouverte": p_d, "p_confirmation": p_c,
                    "p": p_all, "niveau": len(combo),
                })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    confirmed = []
    # Correction pour tests multiples séparée pour les segments simples et les croisements : sinon les
    # centaines de croisements rendent le seuil trop sévère pour les effets simples, les plus utiles.
    for (situation, _), group in table.groupby(["situation", "niveau"]):
        significant = _benjamini_hochberg(group["p"].to_numpy(), fdr)
        for (index, row), keep in zip(group.iterrows(), significant):
            sign = np.sign(row["ecart_pct"])
            strong = abs(row["ecart_pct"]) >= 100 * min_effect
            replicated = all(np.sign(row[c]) == sign and abs(row[c]) >= 50 * min_effect
                             for c in ("ecart_decouverte_pct", "ecart_confirmation_pct"))
            if keep and strong and replicated:
                confirmed.append(index)
    pockets = table.loc[confirmed].copy()
    if pockets.empty:
        return pockets
    # Une poche à deux dimensions n'est gardée que si elle dit plus que ses deux poches simples.
    simple = {(r.situation, tuple(r.dimensions.items())[0]): r.ecart_pct
              for r in pockets[pockets["niveau"] == 1].itertuples()}
    keep = []
    for index, row in pockets.iterrows():
        if row["niveau"] == 1:
            keep.append(index)
            continue
        parents = [simple.get((row["situation"], item)) for item in row["dimensions"].items()]
        parents = [p for p in parents if p is not None]
        if not parents or abs(row["ecart_pct"]) > 1.5 * max(abs(p) for p in parents):
            keep.append(index)
    pockets = pockets.loc[keep]
    pockets["sens"] = np.where(pockets["ecart_pct"] > 0, "spread réel au-dessus de l'estimation",
                               "spread réel en dessous de l'estimation")
    pockets["action"] = [
        f"Pour ce segment, relever l'estimation d'environ {abs(e):.0f} %" if e > 0 else
        f"Pour ce segment, abaisser l'estimation d'environ {abs(e):.0f} %" for e in pockets["ecart_pct"]]
    # Effets simples d'abord (les plus fiables), puis les croisements ; les plus forts en tête.
    pockets["force"] = -pockets["ecart_pct"].abs()
    return pockets.sort_values(["situation", "niveau", "force"]).drop(columns="force").reset_index(drop=True)


# ------------------------------------------------------------------ aiguillage
def routing_table(estimator, readings: pd.DataFrame | None = None, margin: float = 0.05,
                  min_cases: int = 40, seed: int = 7) -> pd.DataFrame:
    """Méthode la plus précise par cas, confirmée sur deux moitiés d'émetteurs.

    `readings` (facultatif) : sortie de `quant.topdown.readings_out_of_fold`, pour
    faire concourir aussi les lectures (entonnoir…).
    """
    cal = estimator.calibration_
    oof = cal["oof"].copy()
    methods = {c: label for c, label in METHODS.items() if c in oof and oof[c].notna().all()}
    if readings is not None:
        oof = oof.merge(readings.drop(columns=["fold", "niveau"], errors="ignore"), on=["row_id", "scheme"], how="left")
        methods.update({c: c.removeprefix("log_") for c in readings.columns if c.startswith("log_")})
    column = {"A": "log_a", "B": "log_b", "C": "log_c", "P": "peer_log", "S": "log_s"}
    oof["log_retenu"] = [row[column[cal["champion"].get(row["key"], "A")]] for _, row in oof.iterrows()]
    df = estimator.df_
    oof["issuer"] = df["issuer"].to_numpy()[oof["row_id"].to_numpy()]
    n_issuer = np.expm1(oof["log_n_issuer"]).round()
    oof["notation"] = np.where(oof["notch"] <= 10, "investissement (BBB- et mieux)", "haut rendement (BB+ et moins)")
    oof["dette"] = np.where(oof["sen_rank"] >= 2, "subordonnée", "senior")
    oof["emetteur"] = np.select([oof["key"] == UNSEEN, n_issuer <= 2], ["inconnu", "1 à 2 obligations"],
                                "3 obligations ou plus")
    rng = np.random.default_rng(seed)
    unique = oof["issuer"].unique()
    half_of = dict(zip(unique, rng.permutation(len(unique)) % 2))
    oof["moitie"] = oof["issuer"].map(half_of)
    q = cal["quantile"]
    oof["score"] = (oof["err"] / oof["sigma"]) if cal["adaptive"] else oof["err"]
    global_q = q["_global"]

    rows = []
    for (rating, debt, issuer), group in oof.groupby(["notation", "dette", "emetteur"]):
        if len(group) < min_cases:
            continue
        errors = {c: float((group[c].astype(float) - group["y_log"]).abs().median()) for c in ["log_retenu", *methods]
                  if group[c].notna().mean() > 0.8}
        halves = [{c: float((h[c].astype(float) - h["y_log"]).abs().median()) for c in errors}
                  for _, h in group.groupby("moitie")]
        current = errors["log_retenu"]
        best = min((c for c in errors if c != "log_retenu"), key=errors.get)
        beats_both = len(halves) == 2 and all(h[best] < (1 - margin) * h["log_retenu"] for h in halves)
        switch = errors[best] < (1 - margin) * current and beats_both
        # Couverture de l'intervalle actuel dans ce cas, et facteur d'élargissement nécessaire.
        cell_keys = group["key"].to_numpy()
        quantiles = np.array([q.get(k, global_q) for k in cell_keys])
        coverage = float((group["score"].to_numpy() <= quantiles).mean())
        needed = float(np.quantile(group["score"] / quantiles, 1 - cal["alpha"]))
        rows.append({
            "notation": rating, "dette": debt, "emetteur": issuer, "cas": len(group),
            "erreur_actuelle_pct": round(100 * current, 2),
            "meilleure_methode": methods.get(best, best), "erreur_meilleure_pct": round(100 * errors[best], 2),
            "changer": bool(switch),
            "methode_recommandee": methods.get(best, best) if switch else "estimation retenue actuelle",
            "couverture_intervalle": round(coverage, 3),
            "facteur_intervalle": round(max(1.0, needed), 2) if coverage < 0.85 else 1.0,
            **{f"erreur_{methods.get(c, c)}_pct": round(100 * v, 2) for c, v in errors.items() if c != "log_retenu"},
        })
    return pd.DataFrame(rows)
