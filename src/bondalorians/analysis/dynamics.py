"""Mouvement du marché entre deux dates de prix.

La base contient les mêmes obligations prises à deux dates. On n'estime que sur une
date (sinon le double d'une obligation est un comparable presque parfait), mais la
seconde renseigne sur la dynamique :

- **mouvement moyen** par notation, secteur, séniorité, maturité, devise : quels
  segments se sont écartés ou resserrés ;
- **part commune** du mouvement : si l'essentiel s'explique par le segment, un spread
  observé à l'autre date peut être recalé par le mouvement de son segment ;
- **obligations atypiques** : mouvement très différent de leur segment (nouvelle
  information sur l'émetteur, ou erreur de donnée) ;
- **stabilité de la courbe des pairs** : erreur quand on l'apprend à une date et qu'on
  l'applique à l'autre (corrigée du mouvement moyen).

Tous les mouvements sont en log : +5 % veut dire spread multiplié par 1,05.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bondalorians.analysis.common import rating_bucket
from bondalorians.analysis.patterns import MATURITY_BINS, MATURITY_LABELS
from bondalorians.data.schema import prepare

DIMENSIONS = {"notation": "Notation", "secteur": "Secteur", "seniorite": "Séniorité", "maturite": "Maturité",
              "devise": "Devise", "industrie": "Industrie"}


def pair_dates(canonical: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Obligations présentes aux deux dates : une ligne par obligation, spreads avant et après."""
    days = pd.to_datetime(canonical["pricing_date"]).dt.normalize()
    unique = sorted(days.dropna().unique())
    if len(unique) < 2:
        raise ValueError("Une seule date de prix : pas de dynamique à mesurer.")
    before, after = pd.Timestamp(unique[0]), pd.Timestamp(unique[-1])
    frames = {}
    for name, day in (("avant", before), ("apres", after)):
        frame = prepare(canonical[days == day].reset_index(drop=True), require_target=True)
        frames[name] = frame.drop_duplicates("isin").set_index("isin")
    common = frames["avant"].index.intersection(frames["apres"].index)
    out = frames["apres"].loc[common].copy()
    out["spread_avant"] = frames["avant"].loc[common, "z_spread"]
    out["mouvement"] = np.log(out["z_spread"] / out["spread_avant"])
    out["notation"] = out["notch"].map(rating_bucket)
    out["secteur"] = out["sector"].astype(str)
    out["seniorite"] = out["seniority"].astype(str)
    out["devise"] = out["currency"].astype(str)
    out["maturite"] = pd.cut(out["maturity_years"], MATURITY_BINS, labels=MATURITY_LABELS).astype(str)
    if "industry" in out.columns:
        out["industrie"] = out["industry"].astype(str)
    return out.reset_index(), before, after


def market_dynamics(canonical: pd.DataFrame, min_bonds: int = 20) -> dict:
    pairs, before, after = pair_dates(canonical)
    move = pairs["mouvement"]
    tables = {}
    for dim, label in DIMENSIONS.items():
        if dim not in pairs.columns:
            continue
        table = (pairs.groupby(dim)
                 .agg(obligations=("mouvement", "size"), emetteurs=("issuer", "nunique"),
                      mouvement_median_pct=("mouvement", lambda v: 100 * np.expm1(np.median(v))),
                      dispersion_pct=("mouvement", lambda v: 100 * np.std(v)))
                 .reset_index().rename(columns={dim: label}))
        table = table[table["obligations"] >= min_bonds].sort_values("mouvement_median_pct")
        tables[dim] = table.round(2)

    # Part commune : mouvement prédit par les segments (notation × séniorité × secteur), hors obligation.
    keys = ["notation", "seniorite", "secteur"]
    groups = pairs.groupby(keys)["mouvement"]
    total, count = groups.transform("sum"), groups.transform("size")
    segment = ((total - move) / (count - 1)).where(count > 1, move.median())
    residual = move - segment
    explained = float(1 - residual.var() / move.var()) if move.var() > 0 else 0.0
    # Part émetteur : le reste est-il commun aux obligations d'un même émetteur ?
    issuer_mean = residual.groupby(pairs["issuer"]).transform("mean")
    issuer_count = pairs.groupby("issuer")["issuer"].transform("size")
    multi = issuer_count > 1
    issuer_share = float(1 - (residual[multi] - issuer_mean[multi]).var() / residual[multi].var()) if multi.sum() > 20 else None

    z = (residual - residual.median()) / (1.4826 * np.median(np.abs(residual - residual.median())) or 1)
    outliers = pairs.assign(ecart_segment_pct=100 * np.expm1(residual), z=z)
    outliers = outliers[np.abs(z) > 4].sort_values("z", key=np.abs, ascending=False)
    outliers = outliers[["isin", "issuer", "rating", "seniority", "sector", "maturity_years", "spread_avant",
                         "z_spread", "ecart_segment_pct"]].head(30).round(2)

    findings = [
        f"{len(pairs)} obligations présentes aux deux dates ({before.date()} et {after.date()}) ; "
        f"mouvement médian {100 * np.expm1(move.median()):+.1f} %, dispersion {100 * move.std():.1f} %.",
        f"Le segment (notation × séniorité × secteur) explique {100 * explained:.0f} % de la variance des mouvements.",
    ]
    if issuer_share is not None:
        findings.append(f"Du reste, {100 * max(issuer_share, 0):.0f} % est commun aux obligations d'un même émetteur "
                        "(nouvelle propre à l'émetteur).")
    if "notation" in tables and len(tables["notation"]):
        t = tables["notation"]
        findings.append("Mouvement par notation : " + ", ".join(
            f"{r.Notation} {r.mouvement_median_pct:+.1f} %" for r in t.itertuples()) + ".")
    findings.append(f"{len(outliers)} obligations ont bougé très différemment de leur segment (plus de 4 écarts "
                    "robustes) : nouvelle sur l'émetteur ou erreur de donnée, à vérifier avant de s'en servir.")
    return {"dates": [str(before.date()), str(after.date())], "findings": findings, "tables": tables,
            "outliers": outliers, "pairs": pairs, "explained_by_segment": explained, "issuer_share": issuer_share}


def curve_stability(canonical: pd.DataFrame, ridge: float = 0.01, extra: tuple = ()) -> dict:
    """Courbe des pairs apprise à la première date, appliquée à la seconde (recalée du mouvement médian).

    Compare à l'erreur hors échantillon à la même date : si l'écart est faible, la structure des
    spreads est stable et les observations de l'autre date sont utilisables après recalage.
    """
    from bondalorians.quant.peer_curve import PeerCurve

    days = pd.to_datetime(canonical["pricing_date"]).dt.normalize()
    unique = sorted(days.dropna().unique())
    first = prepare(canonical[days == unique[0]].reset_index(drop=True), require_target=True)
    last = prepare(canonical[days == unique[-1]].reset_index(drop=True), require_target=True)
    y_last = np.log(last["z_spread"].to_numpy(dtype=float))
    # Mêmes plis par émetteur aux deux dates : les obligations testées ne servent jamais à l'apprentissage,
    # ni à leur date ni à l'autre (sinon leur double ferait la réponse).
    rng = np.random.default_rng(0)
    issuers = np.union1d(first["issuer"].unique(), last["issuer"].unique())
    fold_of = dict(zip(issuers, rng.permutation(len(issuers)) % 5))
    f_first, f_last = first["issuer"].map(fold_of).to_numpy(), last["issuer"].map(fold_of).to_numpy()
    within, across = np.zeros(len(last)), np.zeros(len(last))
    for k in range(5):
        test = f_last == k
        within[test] = PeerCurve(ridge, enriched=True, extra=extra).fit(last[f_last != k]).predict_log(last[test])
        across[test] = PeerCurve(ridge, enriched=True, extra=extra).fit(first[f_first != k]).predict_log(last[test])
    shift = float(np.median(across - y_last))
    across = float(np.median(np.abs(across - shift - y_last)))
    within = float(np.median(np.abs(within - y_last)))
    # Double de l'obligation à l'autre date, recalé : l'erreur qu'on aurait en « trichant ».
    previous = first.drop_duplicates("isin").set_index("isin")["z_spread"]
    twin = np.log(last["isin"].map(previous).to_numpy(dtype=float)) - shift - y_last
    twin = float(np.nanmedian(np.abs(twin)))
    return {
        "erreur_meme_date_pct": round(100 * within, 2),
        "erreur_autre_date_recalee_pct": round(100 * across, 2),
        "recalage_median_pct": round(float(100 * np.expm1(-shift)), 2),
        "erreur_double_autre_date_pct": round(100 * twin, 2),
        "constat": (f"Courbe apprise au {pd.Timestamp(unique[0]).date()} et appliquée au "
                    f"{pd.Timestamp(unique[-1]).date()}, recalée de {100 * np.expm1(-shift):+.1f} % : erreur médiane "
                    f"{100 * across:.1f} %, contre {100 * within:.1f} % hors échantillon à la même date (émetteurs "
                    "retirés). Le spread de la même obligation à l'autre date, recalé, se tromperait de seulement "
                    f"{100 * twin:.1f} % : c'est pourquoi les deux dates ne doivent jamais être mélangées dans un "
                    "backtest. " + ("Structure stable : l'autre date est exploitable après recalage."
                                    if across <= within * 1.1 else
                                    "La structure a changé : ne pas mélanger les dates sans modèle de mouvement.")),
    }
