"""Comprendre l'échantillon : composition, niveaux de spread, relation à la maturité."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bondalorians.analysis.common import BUCKET_ORDER, fmt, pct, r2, rating_bucket, real_bonds


def describe_sample(estimator) -> dict:
    df = real_bonds(estimator).copy()
    df["lettre"] = df["notch"].map(rating_bucket)
    n, issuers = len(df), df["issuer"].nunique()

    per_issuer = df.groupby("issuer").size()
    composition = {
        column: df[column].value_counts().rename_axis("valeur").reset_index(name="n")
        for column in ("lettre", "sector", "currency", "seniority", "country", "coupon_type")
    }
    by_letter = (
        df.groupby("lettre")["z_spread"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .reindex([b for b in BUCKET_ORDER if b in set(df["lettre"])])
        .rename(columns={"count": "n", "10%": "p10", "25%": "p25", "50%": "mediane", "75%": "p75", "90%": "p90"})
        [["n", "p10", "p25", "mediane", "p75", "p90"]]
        .round(1)
        .reset_index()
    )

    # Part de la dispersion des spreads (en log) expliquée par chaque caractéristique seule.
    y = np.log(df["z_spread"].to_numpy())
    explained = []
    for column, label in (("lettre", "Notation (lettre)"), ("rating", "Notation (cran)"),
                          ("seniority", "Séniorité"), ("sector", "Secteur"), ("currency", "Devise"),
                          ("country", "Pays"), ("issuer", "Émetteur")):
        means = pd.Series(y).groupby(df[column].to_numpy()).transform("mean")
        explained.append({"caracteristique": label, "part_expliquee": round(r2(y, means), 3)})
    maturity_bins = pd.qcut(df["maturity_years"], 5, duplicates="drop")
    explained.append({
        "caracteristique": "Maturité (5 tranches)",
        "part_expliquee": round(r2(y, pd.Series(y).groupby(maturity_bins.to_numpy()).transform("mean")), 3),
    })
    explained = pd.DataFrame(explained).sort_values("part_expliquee", ascending=False)

    top = explained.iloc[0]
    letter_share = explained.set_index("caracteristique").loc["Notation (lettre)", "part_expliquee"]
    # Ce que l'émetteur ajoute une fois la notation connue : part du reste qu'il explique.
    residual = y - pd.Series(y).groupby(df["rating"].to_numpy()).transform("mean").to_numpy()
    issuer_extra = r2(residual, pd.Series(residual).groupby(df["issuer"].to_numpy()).transform("mean"))
    single = int((per_issuer == 1).sum())
    findings = [
        f"{fmt(n)} obligations de {fmt(issuers)} émetteurs ; un émetteur a en moyenne "
        f"{fmt(n / issuers, 1)} obligations, et {fmt(single)} émetteurs ({pct(single / issuers)}) n'en ont qu'une.",
        f"Le spread médian est de {fmt(df['z_spread'].median())} pb ; la moitié des obligations sont "
        f"entre {fmt(df['z_spread'].quantile(.25))} et {fmt(df['z_spread'].quantile(.75))} pb.",
        f"{top['caracteristique']} est la caractéristique qui explique le plus la dispersion des spreads, "
        f"à elle seule ({pct(top['part_expliquee'])}).",
        f"La lettre de notation seule explique {pct(letter_share)} de la dispersion. Une fois la notation connue, "
        f"l'émetteur explique encore {pct(issuer_extra)} de ce qui reste : c'est pourquoi l'estimation suit la "
        "courbe propre à chaque émetteur quand elle existe.",
    ]
    big = composition["lettre"].iloc[0]
    findings.append(
        f"La catégorie la plus représentée est {big['valeur']} ({pct(big['n'] / n)} de l'échantillon) : "
        "les estimations y reposent sur plus de comparables que dans les catégories rares."
    )
    dropped = estimator.dropped_
    if dropped:
        findings.append(f"{fmt(dropped)} ligne(s) de la base ont été écartées (notation inconnue, maturité ou spread non positif).")

    points = df[["isin", "issuer", "rating", "lettre", "notch", "seniority", "sector", "currency",
                 "maturity_years", "z_spread"]]
    return {
        "title": "Échantillon",
        "findings": findings,
        "tables": {
            "taille": {"obligations": n, "emetteurs": issuers, "lignes_ecartees": dropped,
                       "obligations_par_emetteur_median": float(per_issuer.median()),
                       "part_du_reste_expliquee_par_emetteur": round(issuer_extra, 3)},
            "spread_par_lettre": by_letter,
            "part_expliquee": explained,
            "composition": {k: v for k, v in composition.items()},
        },
        "points": points,
    }
