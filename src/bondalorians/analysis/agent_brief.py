"""Fiche de connaissances pour des agents : ce que la base et le backtest ont appris.

Un seul objet JSON, lisible par un agent :

- `structure` : ce qui détermine les spreads, poids de l'émetteur, primes de subordination ;
- `aiguillage` : par cas (notation, séniorité, information sur l'émetteur), la méthode à
  utiliser et le facteur d'élargissement de l'intervalle ;
- `poches` : segments où l'estimation se trompe systématiquement, confirmés sur des
  émetteurs indépendants ;
- `colonnes_supplementaires` : ESG, ancienneté, duration… utiles ou non ;
- `regles` : les consignes qui en découlent, en phrases, chacune avec sa preuve chiffrée.

`patterns_for_bond(brief, bond)` renvoie ce qui s'applique à une obligation donnée.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from bondalorians.analysis.common import jsonable, rating_bucket, real_bonds
from bondalorians.analysis.patterns import (
    AGE_BINS, AGE_LABELS, DIMENSIONS, MATURITY_BINS, MATURITY_LABELS, _tier, find_pockets, routing_table,
    segment_edges,
)
from bondalorians.analysis.sample import describe_sample

HOW_TO_USE = (
    "Pour estimer une obligation : 1) chercher sa case dans `aiguillage` et utiliser la méthode recommandée, en "
    "multipliant la demi-largeur de l'intervalle par `facteur_intervalle` ; 2) vérifier les `poches` qui la "
    "concernent et mentionner l'ajustement suggéré ; 3) ne jamais citer un chiffre absent de cette fiche ou des "
    "outils. Toutes les erreurs sont hors échantillon (obligations estimées sans avoir servi à l'apprentissage). "
    "Les poches sont des tendances moyennes d'un segment, pas des certitudes pour une obligation."
)


def build_brief(estimator, readings: pd.DataFrame | None = None, include_extra: bool = True,
                source: str = "", max_pockets: int = 25) -> dict:
    if estimator.calibration_ is None:
        raise RuntimeError("Estimateur non calibré.")
    df = real_bonds(estimator)
    sample = describe_sample(estimator)
    explained = sample["tables"]["part_expliquee"]
    pockets = find_pockets(estimator, "retenu")
    curve_pockets = find_pockets(estimator, "courbe")
    routing = routing_table(estimator, readings=readings)

    sub = estimator.subordination_
    structure = {
        "obligations": len(df), "emetteurs": int(df["issuer"].nunique()),
        "obligations_par_emetteur_mediane": float(df.groupby("issuer").size().median()),
        "part_du_spread_expliquee_par_variable": {
            r.caracteristique: round(r.part_expliquee, 3) for r in explained.itertuples()},
        "poids_emetteur_k": round(estimator.k_, 3),
        "poids_emetteur_avec_1_3_10_obligations": [round(n / (n + estimator.k_), 2) for n in (1, 3, 10)],
        "dispersion_entre_obligations_d_un_emetteur": estimator.shrinkage_.get("noise_sd"),
        "dispersion_entre_emetteurs": estimator.shrinkage_.get("issuer_sd"),
        "primes_de_subordination_pct": {} if sub is None else {
            k: round(100 * (math.exp(v) - 1), 1) for k, v in sub.categories_.items()},
        "modele_retenu_par_palier": estimator.calibration_["champion"],
    }
    extra = None
    if include_extra:
        from bondalorians.analysis.extra_columns import evaluate_extra_columns

        if any(c in df.columns for c in ("esg_score", "age_years", "duration", "green_flag", "holdco_flag",
                                          "call_ratio")):
            result = evaluate_extra_columns(df)
            extra = {"recommandees": result["recommended"], "constats": result["findings"],
                     "detail": result["tables"]["variables"]}

    rules = _rules(structure, routing, pockets, curve_pockets, extra, max_pockets)
    brief = {
        "meta": {
            "genere_le": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "source": source,
            "version_modele": estimator.model_version_, "seuils_segments": segment_edges(estimator.df_),
            "mode_d_emploi": HOW_TO_USE,
        },
        "structure": structure,
        "aiguillage": routing,
        "poches": pockets.head(max_pockets),
        "poches_courbe_des_pairs": curve_pockets.head(max_pockets),
        "colonnes_supplementaires": extra,
        "regles": rules,
    }
    return jsonable(brief)


def _rules(structure, routing, pockets, curve_pockets, extra, max_pockets) -> list[dict]:
    rules = []

    def add(kind, text, evidence, condition=None):
        rules.append({"id": f"R{len(rules) + 1:02d}", "type": kind, "consigne": text, "preuve": evidence,
                      "condition": condition or {}})

    k = structure["poids_emetteur_k"]
    weights = structure["poids_emetteur_avec_1_3_10_obligations"]
    add("structure",
        ("Les obligations d'un même émetteur sont cohérentes : s'appuyer fortement sur la courbe de l'émetteur."
         if k < 1.5 else
         "Les obligations d'un même émetteur sont dispersées : ne pas se fier à une ou deux obligations de l'émetteur."),
        f"k = {k} ; poids de l'émetteur avec 1, 3 et 10 obligations : {weights}.")
    shares = {k: v for k, v in structure["part_du_spread_expliquee_par_variable"].items()
              if k != "Notation (lettre)"}  # doublon de la notation en crans
    top = sorted(shares.items(), key=lambda kv: -kv[1])[:3]
    add("structure", "Variables qui déterminent le plus le spread : " + ", ".join(k for k, _ in top) + ".",
        "Parts de la dispersion expliquées seules : " + ", ".join(f"{k} {100 * v:.0f} %" for k, v in top) + ".")
    for row in routing.itertuples():
        condition = {"notation": row.notation, "dette": row.dette, "emetteur": row.emetteur}
        if row.changer:
            add("aiguillage", f"Obligation {row.notation}, dette {row.dette}, émetteur {row.emetteur} : utiliser "
                f"{row.methode_recommandee}.",
                f"Erreur médiane {row.erreur_meilleure_pct} % contre {row.erreur_actuelle_pct} % ({row.cas} cas), "
                "écart confirmé sur deux moitiés d'émetteurs.", condition)
        if row.facteur_intervalle > 1:
            add("intervalle", f"Obligation {row.notation}, dette {row.dette}, émetteur {row.emetteur} : élargir "
                f"l'intervalle d'un facteur {row.facteur_intervalle}.",
                f"Couverture mesurée {100 * row.couverture_intervalle:.0f} % au lieu de 90 % ({row.cas} cas).", condition)
    if pockets.empty:
        add("poche", "Aucune poche de biais confirmée : appliquer l'aiguillage sans ajustement par segment.",
            "Tous les segments d'au moins 30 obligations et 8 émetteurs ont été testés, avec confirmation sur des "
            "émetteurs indépendants.")
    for row in pockets.head(max_pockets).itertuples():
        add("poche", f"{row.situation.capitalize()}, {row.segment} : {row.action.lower()}.",
            f"Écart médian {row.ecart_pct:+.1f} % ({row.obligations} obligations, {row.emetteurs} émetteurs) ; "
            f"{row.ecart_decouverte_pct:+.1f} % puis {row.ecart_confirmation_pct:+.1f} % sur deux moitiés d'émetteurs.",
            {**row.dimensions, "situation": row.situation})
    seen_segments = set()
    for row in curve_pockets.itertuples():
        if row.segment in seen_segments or len(seen_segments) >= 5:
            continue
        seen_segments.add(row.segment)
        add("piste", f"Effet non capté par la courbe des pairs : {row.segment} ({row.sens}). Variable ou interaction "
            "à envisager.", f"Écart médian {row.ecart_pct:+.1f} % ({row.obligations} obligations).", row.dimensions)
    if extra:
        add("colonnes", "Colonnes supplémentaires à utiliser : " + (", ".join(extra["recommandees"]) or "aucune") + ".",
            " ".join(extra["constats"][1:4]))
    add("garde-fou", "Ne jamais utiliser le prix ni le rendement de l'obligation estimée pour estimer son spread.",
        "Ils sont liés mécaniquement au spread : les utiliser reviendrait à recalculer la cotation.")
    return rules


def _bond_segments(brief: dict, bond: dict) -> dict:
    edges = brief["meta"]["seuils_segments"]
    maturity = float(bond["maturity_years"])
    values = {
        "notation": rating_bucket(bond["notch"]), "seniorite": str(bond["seniority"]), "secteur": str(bond["sector"]),
        "devise": str(bond["currency"]), "pays": str(bond.get("country", "NA")),
        "maturite": str(pd.cut([maturity], MATURITY_BINS, labels=MATURITY_LABELS)[0]),
        "taille": _tier([bond.get("log_amt")], edges["taille"], ["petite", "moyenne", "grande"]).iloc[0],
        "remboursable": "oui" if bond.get("callable") else "non", "coupon": str(bond.get("coupon_type", "FIXED")),
    }
    if "esg" in edges and bond.get("esg_score") is not None and not pd.isna(bond.get("esg_score")):
        values["esg"] = _tier([bond["esg_score"]], edges["esg"], ["ESG faible", "ESG moyen", "ESG élevé"]).iloc[0]
    if "anciennete" in edges and bond.get("age_years") is not None and not pd.isna(bond.get("age_years")):
        values["anciennete"] = str(pd.cut([float(bond["age_years"])], AGE_BINS, labels=AGE_LABELS)[0])
    for column, name in (("industry", "industrie"), ("structure", "structure")):
        if bond.get(column) is not None:
            values[name] = str(bond[column])
    if bond.get("green_flag") is not None and not pd.isna(bond.get("green_flag")):
        values["vert"] = "oui" if bond["green_flag"] == 1 else "non"
    return values


def patterns_for_bond(brief: dict, bond: dict, issuer_bonds: int | None) -> dict:
    """Case d'aiguillage, poches et règles qui s'appliquent à une obligation préparée.

    `issuer_bonds` : nombre d'autres obligations de l'émetteur dans la base (None = émetteur inconnu).
    """
    values = _bond_segments(brief, bond)
    case = {
        "notation": "investissement (BBB- et mieux)" if bond["notch"] <= 10 else "haut rendement (BB+ et moins)",
        "dette": "subordonnée" if bond["sen_rank"] >= 2 else "senior",
        "emetteur": "inconnu" if not issuer_bonds else ("1 à 2 obligations" if issuer_bonds <= 2 else
                                                        "3 obligations ou plus"),
    }
    situation = "émetteur inconnu" if not issuer_bonds else "émetteur connu"
    routing = [r for r in brief["aiguillage"] if all(r[k] == v for k, v in case.items())]
    pockets = [p for p in brief["poches"] if p["situation"] == situation
               and all(values.get(d) == v for d, v in p["dimensions"].items())]
    rules = [r for r in brief["regles"] if r["type"] in ("aiguillage", "intervalle")
             and all(r["condition"].get(k) == v for k, v in case.items())]
    rules += [r for r in brief["regles"] if r["type"] == "poche"
              and r["condition"].get("situation") == situation
              and all(values.get(d) == v for d, v in r["condition"].items() if d != "situation")]
    return jsonable({"segments": values, "case": case, "situation": situation,
                     "aiguillage": routing[0] if routing else None, "poches": pockets, "regles": rules})


def save_brief(brief: dict, path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(brief, handle, ensure_ascii=False, indent=2, default=str)
