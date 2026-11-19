"""Comprendre chaque modèle : ce qu'il a appris, comment il se trompe."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bondalorians.analysis.common import TIER_LABELS, fmt, notch_label, pct, r2, real_bonds
from bondalorians.data.schema import to_notch
from bondalorians.analysis.forest import fit_forest, forest_out_of_fold, grouped_permutation_importance
from bondalorians.quant.gbm import _features as catboost_features

LETTERS = {"AA": 3, "A": 6, "BBB": 9, "BB": 12, "B": 15}
MODEL_NAMES = {
    "peer_log": "Courbe des pairs", "log_a": "A · Comparables", "log_b": "B · CatBoost",
    "log_c": "C · Comparables corrigés", "log_s": "S · Ancrage senior", "log_rf": "Forêt aléatoire",
}
SHAP_LABELS = {
    "notch": "Notation", "log_t": "Maturité", "sen_rank": "Rang de séniorité", "log_amt": "Taille",
    "callable_num": "Remboursable", "sector": "Secteur", "country": "Pays", "currency": "Devise",
    "coupon_type": "Type de coupon", "seniority": "Séniorité (catégorie)",
}


def _reference_bond(df: pd.DataFrame) -> dict:
    """Obligation « typique » : modalité la plus fréquente, valeurs médianes."""
    mode = {c: df[c].mode().iat[0] for c in ("sector", "currency", "coupon_type", "country", "seniority")}
    return {**mode, "callable": False, "log_amt": float(df["log_amt"].median()),
            "sen_rank": float(df.loc[df["seniority"] == mode["seniority"], "sen_rank"].iat[0])}


# ------------------------------------------------------------- courbe des pairs
def peer_curve_view(estimator) -> dict:
    df = real_bonds(estimator)
    ref = _reference_bond(df)
    maturities = np.exp(np.linspace(np.log(0.5), np.log(30), 40))
    grid = pd.DataFrame([
        {**ref, "lettre": letter, "notch": float(notch), "maturity_years": t, "log_t": float(np.log(t))}
        for letter, notch in LETTERS.items() for t in maturities
    ])
    grid["z_spread"] = np.exp(estimator.peer_.predict_log(grid))

    fitted = estimator.peer_.predict_log(df)
    in_sample = r2(np.log(df["z_spread"]), fitted)

    def at(letter, t):
        row = {**ref, "notch": float(LETTERS[letter]), "log_t": float(np.log(t))}
        return float(np.exp(estimator.peer_.predict_log(pd.DataFrame([row]))[0]))

    findings = [
        f"La courbe des pairs explique {pct(in_sample)} de la dispersion des spreads (en log) à elle seule, "
        "sans rien savoir de l'émetteur.",
        f"À 5 ans, pour une obligation {ref['seniority']} en {ref['currency']} du secteur {ref['sector']} : "
        f"{fmt(at('A', 5))} pb en A, {fmt(at('BBB', 5))} pb en BBB, {fmt(at('BB', 5))} pb en BB.",
        f"Pour un BBB, passer de 2 à 10 ans fait passer le spread de {fmt(at('BBB', 2))} à {fmt(at('BBB', 10))} pb.",
        "Les courbes sont tracées pour une obligation typique de l'échantillon : changer la séniorité, "
        "le secteur ou la devise les décale vers le haut ou le bas.",
    ]
    points = df[["issuer", "rating", "notch", "maturity_years", "z_spread"]].copy()
    points["lettre"] = points["notch"].map(
        lambda n: next((k for k, v in LETTERS.items() if abs(n - v) <= 1), None)
    )
    return {
        "title": "Courbe des pairs",
        "findings": findings,
        "tables": {"r2_log": round(in_sample, 3), "obligation_type": ref},
        "points": {"curves": grid[["lettre", "maturity_years", "z_spread"]],
                   "bonds": points.dropna(subset=["lettre"])},
    }


# ---------------------------------------------------------- écart émetteur
def shrinkage_view(estimator) -> dict:
    df = real_bonds(estimator)
    resid = pd.Series(np.log(df["z_spread"].to_numpy()) - estimator.peer_.predict_log(df), index=df.index)
    by_issuer = resid.groupby(df["issuer"]).agg(["mean", "size"]).rename(columns={"mean": "ecart_brut", "size": "n"})
    k = estimator.k_
    by_issuer["poids"] = by_issuer["n"] / (by_issuer["n"] + k)
    by_issuer["ecart_retenu"] = by_issuer["poids"] * by_issuer["ecart_brut"]
    by_issuer = by_issuer.reset_index()

    info = estimator.shrinkage_
    findings = [f"k vaut {fmt(k, 2)} : avec 1 obligation, l'écart de l'émetteur est retenu à "
                f"{pct(1 / (1 + k))} ; avec 3, à {pct(3 / (3 + k))} ; avec 10, à {pct(10 / (10 + k))}."]
    if info.get("estimated"):
        findings.append(
            f"Deux obligations d'un même émetteur s'écartent typiquement de ±{pct(info['noise_sd'])} entre elles ; "
            f"un émetteur s'écarte typiquement de ±{pct(info['issuer_sd'])} de la courbe des pairs. "
            "Plus le second chiffre domine, plus on fait confiance à l'émetteur, même avec peu d'obligations."
        )
    extreme = by_issuer.loc[by_issuer["ecart_retenu"].abs().idxmax()]
    findings.append(
        f"L'écart retenu le plus fort est celui de {extreme['issuer']} : {fmt(100 * extreme['ecart_retenu'], 0)} % "
        f"par rapport à ses pairs, sur {int(extreme['n'])} obligation(s)."
    )
    return {
        "title": "Écart propre à l'émetteur",
        "findings": findings,
        "tables": {"k": round(k, 3), **{key: info.get(key) for key in ("noise_sd", "issuer_sd", "estimated")}},
        "points": by_issuer,
    }


# ------------------------------------------------------------- comparables
def tuning_view(estimator) -> dict:
    report = estimator.calibration_report()
    tuning = report["comparables_tuning"]
    defaults = estimator.settings["comparables"]
    labels = {"notch": "Notation", "maturity": "Maturité", "sector": "Secteur", "country": "Pays", "size": "Taille"}
    rows = [{"critere": labels[key], "reglage": "Par défaut", "poids": defaults["weights"][key]} for key in labels]
    rows += [{"critere": labels[key], "reglage": "Appliqué", "poids": estimator.params_["weights"][key]} for key in labels]
    applied = estimator.params_
    if tuning is None:
        findings = ["Le réglage automatique est désactivé : les poids par défaut sont appliqués."]
    else:
        valid = tuning["valid_error_pct"]
        findings = [
            f"{tuning['trials']} combinaisons essayées sur {tuning['n_tune_cases']} obligations ; "
            f"sur {tuning['n_valid_cases']} autres obligations, l'erreur typique passe de {fmt(valid['default'], 1)} % "
            f"à {fmt(valid['tuned'], 1)} %.",
            "Réglage adopté." if tuning["adopted"] else "Réglage rejeté : il ne faisait pas mieux hors réglage.",
        ]
    top = max(applied["weights"], key=applied["weights"].get)
    findings.append(
        f"Critère le plus pesant : {labels[top].lower()} ({pct(applied['weights'][top])}). "
        f"{applied['top_k']} comparables au plus, largeur de noyau {fmt(applied['bandwidth'], 2)}."
    )
    findings.append(
        "Ces poids varient beaucoup d'un jeu de données à l'autre : ils mesurent ce qui aide à prédire ici, "
        "pas une vérité de marché."
    )
    return {"title": "Choix des comparables", "findings": findings,
            "tables": {"parametres": applied, "reglage": tuning}, "points": pd.DataFrame(rows)}


def comparables_view(estimator, result: dict) -> dict:
    df = real_bonds(estimator)
    same = df[(df["currency"] == result["bond"]["currency"])]
    comps = pd.DataFrame(result["comparables"])
    target_notch = to_notch(result["bond"]["rating"])
    findings = []
    if not comps.empty:
        lo, hi = comps["adjusted_z_spread_bp"].quantile([0.1, 0.9])
        top_share = comps["weight"].nlargest(5).sum() / comps["weight"].sum()
        findings += [
            f"{len(comps)} comparables retenus ; les 5 plus proches pèsent {pct(top_share)} du total.",
            f"Une fois ramenés aux caractéristiques de votre obligation, 80 % d'entre eux donnent entre "
            f"{fmt(lo)} et {fmt(hi)} pb.",
        ]
    offset = result["issuer_offset"]
    if offset["n_bonds"]:
        findings.append(
            f"L'émetteur a {offset['n_bonds']} autre(s) obligation(s) : sa propre courbe pèse "
            f"{pct(offset['lambda'])}, les comparables {pct(1 - offset['lambda'])}."
        )
    else:
        findings.append("Émetteur absent de l'univers : l'estimation repose entièrement sur les comparables.")
    return {"title": "Comparables de votre obligation", "findings": findings,
            "tables": {"filtre": result["filter_level"]},
            "points": {"universe": same[["isin", "issuer", "rating", "notch", "maturity_years", "z_spread"]],
                       "comparables": comps, "target_notch": target_notch}}


# ----------------------------------------------------------------- CatBoost
def catboost_view(estimator, sample: int = 1500, seed: int = 7) -> dict:
    if estimator.ml_ is None:
        return {"title": "CatBoost (modèle B)", "available": False,
                "findings": ["CatBoost est désactivé (BONDALORIANS_USE_ML=0) ou non installé."],
                "tables": {}, "points": None}
    from catboost import Pool

    from bondalorians.quant.gbm import CATEGORICAL

    df = real_bonds(estimator)
    df = df.sample(min(sample, len(df)), random_state=seed)
    x = catboost_features(df)
    values = estimator.ml_.model_.get_feature_importance(Pool(x, cat_features=CATEGORICAL), type="ShapValues")
    shap = pd.DataFrame(values[:, :-1], columns=x.columns)
    importance = (shap.abs().mean() * 100).rename(index=SHAP_LABELS).sort_values(ascending=False)
    importance = importance.rename("effet_moyen_pct").reset_index().rename(columns={"index": "variable"})
    dependence = pd.DataFrame({
        "notch": df["notch"].to_numpy(), "rating": df["rating"].to_numpy(),
        "effet_pct": 100 * (np.exp(shap["notch"].to_numpy()) - 1),
    })
    by_notch = dependence.groupby("notch")["effet_pct"].median()
    top = importance.head(3)
    findings = [
        "Effet moyen = de combien, en moyenne, une variable déplace le spread prédit d'une obligation "
        "par rapport à l'obligation moyenne.",
        "Les trois variables les plus utilisées : " + ", ".join(
            f"{r.variable.lower()} ({fmt(r.effet_moyen_pct)} %)" for r in top.itertuples()) + ".",
    ]
    if len(by_notch) > 2:
        lo, hi = by_notch.index.min(), by_notch.index.max()
        findings.append(
            f"Effet de la notation seule : {fmt(by_notch.loc[lo])} % en {notch_label(lo)}, "
            f"{fmt(by_notch.loc[hi])} % en {notch_label(hi)}. La contrainte de monotonie garantit que, pour une "
            "obligation donnée, dégrader la notation ne baisse jamais le spread prédit ; les moyennes par notation "
            "peuvent légèrement fluctuer là où les obligations sont rares."
        )
    return {"title": "CatBoost (modèle B)", "available": True, "findings": findings,
            "tables": {"importance": importance}, "points": {"importance": importance, "dependence": dependence}}


def corrector_view(estimator) -> dict:
    oof = (estimator.calibration_ or {}).get("oof")
    if oof is None or oof["log_c"].isna().all():
        return {"title": "Correcteur (modèle C)", "available": False,
                "findings": ["Le correcteur est désactivé ou CatBoost n'est pas installé."], "tables": {}, "points": None}
    corr = 100 * (np.exp(oof["log_c"].astype(float) - oof["log_a"]) - 1)
    points = pd.DataFrame({"correction_pct": corr, "palier": oof["key"].map(TIER_LABELS)})
    err_a = (oof["log_a"] - oof["y_log"]).abs()
    err_c = (oof["log_c"].astype(float) - oof["y_log"]).abs()
    findings = [
        f"Le correcteur modifie l'estimation du modèle A de {fmt(corr.abs().median(), 1)} % en médiane ; "
        f"90 % des corrections sont entre {fmt(corr.quantile(.05), 1)} % et {fmt(corr.quantile(.95), 1)} %.",
        f"Hors échantillon, il améliore le modèle A dans {pct((err_c < err_a).mean())} des cas.",
        f"Erreur médiane : {fmt(100 * err_a.median(), 1)} % pour A, {fmt(100 * err_c.median(), 1)} % pour C.",
    ]
    return {"title": "Correcteur (modèle C)", "available": True, "findings": findings,
            "tables": {}, "points": points}


# --------------------------------------------- effets comparés et comparaison
def effect_curves(estimator, forest=None, sample: int = 300, seed: int = 7) -> dict:
    """Effet moyen de la notation et de la maturité, selon chaque modèle (dépendance partielle)."""
    df = real_bonds(estimator).sample(min(sample, len(estimator.df_)), random_state=seed)
    encoder, model = forest if forest is not None else fit_forest(estimator, seed)
    rows = []
    for variable, values in (("notch", np.arange(2, 18)), ("log_t", np.log(np.array([1, 2, 3, 5, 7, 10, 15, 20, 30])))):
        for value in values:
            frame = df.copy()
            frame[variable] = float(value)
            predictions = {"Courbe des pairs": estimator.peer_.predict_log(frame),
                           "Forêt aléatoire": model.predict(encoder.transform(frame))}
            if estimator.ml_ is not None:
                predictions["B · CatBoost"] = estimator.ml_.predict_log(frame)
            for name, log_pred in predictions.items():
                rows.append({"variable": variable, "valeur": float(value), "modele": name,
                             "z_spread": float(np.exp(log_pred).mean())})
    points = pd.DataFrame(rows)
    points["axe"] = np.where(points["variable"] == "notch", points["valeur"], np.exp(points["valeur"]))
    findings = [
        "Chaque courbe montre le spread moyen prédit si l'on change une seule caractéristique de toutes les "
        "obligations, le reste restant tel quel.",
        "Courbes proches : les modèles ont appris la même chose. Une forêt aléatoire produit des paliers, "
        "un modèle de régression des courbes lisses.",
    ]
    notch_rows = points[points["variable"] == "notch"]
    for name, group in notch_rows.groupby("modele"):
        series = group.sort_values("valeur")["z_spread"].to_numpy()
        drops = int((np.diff(series) < -1).sum())
        if drops:
            findings.append(f"{name} : le spread baisse {drops} fois quand la notation se dégrade, "
                            "ce qu'un analyste crédit refuserait.")
    return {"title": "Effets comparés", "findings": findings, "tables": {}, "points": points}


def model_comparison(estimator, rf_oof: pd.DataFrame | None = None) -> dict:
    """Erreur hors échantillon de chaque modèle, par palier, sur les mêmes obligations."""
    oof = estimator.calibration_["oof"].copy()
    if rf_oof is not None:
        oof = oof.merge(rf_oof, on=["row_id", "scheme"], how="left")
    truth = np.exp(oof["y_log"])
    rows = []
    for column, name in MODEL_NAMES.items():
        if column not in oof or oof[column].isna().all():
            continue
        error_bp = (np.exp(oof[column].astype(float)) - truth).abs()
        error_rel = (oof[column].astype(float) - oof["y_log"]).abs()
        for key, group in oof.groupby("key"):
            rows.append({"modele": name, "palier": TIER_LABELS[key], "n": len(group),
                         "erreur_mediane_bp": round(float(error_bp[group.index].median()), 1),
                         "erreur_mediane_pct": round(float(100 * error_rel[group.index].median()), 1)})
    table = pd.DataFrame(rows)
    champion = estimator.calibration_["champion"]
    findings = ["Erreur médiane = la moitié des obligations sont estimées avec une erreur plus faible. "
                "Chaque obligation est estimée sans avoir servi à l'apprentissage."]
    for tier, group in table.groupby("palier"):
        best = group.loc[group["erreur_mediane_pct"].idxmin()]
        findings.append(f"{tier} : meilleur modèle {best['modele']} ({fmt(best['erreur_mediane_bp'], 1)} pb).")
    if rf_oof is not None and "A · Comparables" in set(table["modele"]):
        pivot = table.pivot(index="palier", columns="modele", values="erreur_mediane_pct")
        wins = int((pivot["Forêt aléatoire"] < pivot["A · Comparables"]).sum())
        findings.append(
            f"La forêt aléatoire bat le modèle A dans {wins} palier(s) sur {len(pivot)}. Elle n'utilise pas "
            "la courbe de l'émetteur : elle ne peut gagner que là où l'émetteur apporte peu."
        )
    findings.append("Modèle retenu par palier dans l'estimation : " + ", ".join(
        f"{TIER_LABELS[k]} → {v}" for k, v in champion.items()) + ".")
    return {"title": "Comparaison des modèles", "findings": findings, "tables": {"erreurs": table}, "points": table}


def forest_view(estimator) -> dict:
    importance = grouped_permutation_importance(estimator)
    top = importance.head(3)
    findings = [
        "Importance = de combien l'erreur de la forêt augmente quand on brouille une variable "
        "(mesuré sur des émetteurs qu'elle n'a jamais vus).",
        "Variables dont la forêt dépend le plus : " + ", ".join(
            f"{r.variable.lower()} (+{fmt(r.hausse_erreur_pct, 1)} points d'erreur)" for r in top.itertuples()) + ".",
    ]
    useless = importance[importance["hausse_erreur_pct"] <= 0.05]["variable"].tolist()
    if useless:
        findings.append("Variables quasi inutiles pour elle : " + ", ".join(v.lower() for v in useless) + ".")
    return {"title": "Forêt aléatoire", "findings": findings, "tables": {"importance": importance}, "points": importance}


def confidence_view(estimator) -> dict:
    cal = estimator.calibration_
    oof = cal["oof"].copy()
    oof["decile"] = pd.qcut(oof["sigma"].rank(method="first"), 10, labels=False) + 1
    oof["score"] = 100 * (1 - (oof["decile"] - 0.5) / 10)
    by_decile = oof.groupby("decile").agg(
        score=("score", "first"), erreur_mediane_pct=("err", lambda e: 100 * e.median()),
        erreur_prevue_pct=("sigma", lambda s: 100 * s.median()),
    ).reset_index()
    from bondalorians.analysis.quality import honest_coverage

    oof["issuer"] = estimator.df_["issuer"].to_numpy()[oof["row_id"].to_numpy()]
    rows = []
    for key, group in oof.groupby("key"):
        q = cal["quantile"].get(key, cal["quantile"]["_global"])
        half = q * group["sigma"] if cal["adaptive"] else q
        # Couverture mesurée sur des émetteurs qui n'ont pas servi à calibrer l'intervalle.
        rows.append({"palier": TIER_LABELS[key], "n": len(group),
                     "couverture": round(honest_coverage(group, cal), 3),
                     "largeur_mediane_pct": round(float(100 * (np.exp(np.median(half)) - np.exp(-np.median(half)))), 1)})
    coverage = pd.DataFrame(rows)
    easiest, hardest = by_decile.iloc[0], by_decile.iloc[-1]
    rank = oof["sigma"].corr(oof["err"], method="spearman")
    findings = [
        f"Les 10 % de cas jugés les plus faciles ont une erreur médiane de {fmt(easiest['erreur_mediane_pct'], 1)} % ; "
        f"les 10 % les plus difficiles, {fmt(hardest['erreur_mediane_pct'], 1)} %.",
        f"Le classement par difficulté est {'utile' if rank > 0.2 else 'faible' if rank > 0.05 else 'sans valeur'} "
        f"(corrélation de rang {fmt(rank, 2)} entre difficulté prévue et erreur réalisée).",
        "Couverture visée : 90 %. Obtenue, sur des émetteurs non utilisés pour calibrer l'intervalle : " + ", ".join(
            f"{r.palier.lower()} {pct(r.couverture)}" for r in coverage.itertuples()) + ".",
    ]
    return {"title": "Confiance et intervalle", "findings": findings,
            "tables": {"par_decile": by_decile, "couverture": coverage},
            "points": {"deciles": by_decile, "coverage": coverage}}


def all_model_views(estimator, include_forest: bool = True) -> dict:
    """Toutes les vues ; la forêt (plus lente) est optionnelle."""
    forest_oof = forest_out_of_fold(estimator) if include_forest else None
    views = {
        "courbe": peer_curve_view(estimator), "emetteur": shrinkage_view(estimator),
        "reglage": tuning_view(estimator), "catboost": catboost_view(estimator),
        "correcteur": corrector_view(estimator), "comparaison": model_comparison(estimator, forest_oof),
        "confiance": confidence_view(estimator),
    }
    if include_forest:
        views["foret"] = forest_view(estimator)
        views["effets"] = effect_curves(estimator)
    return views
