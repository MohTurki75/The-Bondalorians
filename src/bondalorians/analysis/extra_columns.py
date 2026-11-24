"""Les colonnes supplémentaires (ESG, ancienneté, duration, prix, rendement) aident-elles ?

Pour chaque variable candidate, on compare l'erreur hors échantillon de la
courbe des pairs avec et sans elle, sur les mêmes plis (émetteurs entiers
retirés, puis obligations retirées au hasard). Une variable n'est
recommandée que si elle réduit l'erreur de façon nette dans les deux cas.

Le prix et le rendement (YTM) ne sont jamais candidats : ils sont liés
mécaniquement au spread (rendement ≈ taux sans risque + spread ; prix
fonction du rendement). Les utiliser reviendrait à recalculer le spread de la
cotation au lieu de l'estimer. On mesure seulement à quel point ils le
« devinent », pour le montrer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bondalorians.quant.peer_curve import EXTRA_FEATURES, PeerCurve

LABELS = {"esg": "ESG", "age": "Ancienneté de l'émission", "duration": "Duration", "vert": "Obligation verte",
          "holdco": "Dette de holding (structure)", "call": "Call avant l'échéance (part de vie restante)",
          "bailin": "Dette bail-in"}
# Une duration modifiée (ou effective) se calcule au rendement de marché, donc avec le spread :
# elle en contient une partie. Son gain apparent peut être une fuite ; jamais recommandée d'office.
LEAKY = {"duration"}
MIN_GAIN = 0.01  # 1 % d'erreur en moins, dans chaque situation


def _folds(df: pd.DataFrame, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    issuers = df["issuer"].unique()
    fold_of = dict(zip(issuers, rng.permutation(len(issuers)) % 5))
    return {"émetteur inconnu": df["issuer"].map(fold_of).to_numpy(), "émetteur connu": rng.permutation(len(df)) % 5}


def _oof_error(df: pd.DataFrame, folds: np.ndarray, extra: tuple, ridge: float) -> float:
    y = np.log(df["z_spread"].to_numpy(dtype=float))
    prediction = np.zeros(len(df))
    for k in np.unique(folds):
        train, test = folds != k, folds == k
        curve = PeerCurve(ridge, enriched=True, extra=extra).fit(df[train])
        prediction[test] = curve.predict_log(df[test])
    return float(np.median(np.abs(prediction - y)))


def evaluate_extra_columns(df: pd.DataFrame, ridge: float = 0.01) -> dict:
    """`df` : univers préparé (schema.prepare). Renvoie tableaux et constats."""
    folds = _folds(df)
    available = [name for name, column in EXTRA_FEATURES.items()
                 if column in df.columns and pd.to_numeric(df[column], errors="coerce").notna().mean() > 0.5]
    base = {situation: _oof_error(df, f, (), ridge) for situation, f in folds.items()}
    y = np.log(df["z_spread"].to_numpy(dtype=float))
    resid = y - PeerCurve(ridge, enriched=True).fit(df).predict_log(df)

    rows, kept = [], []
    for name in available:
        values = pd.to_numeric(df[EXTRA_FEATURES[name]], errors="coerce")
        with_feature = {situation: _oof_error(df, f, (name,), ridge) for situation, f in folds.items()}
        gains = {s: 1 - with_feature[s] / base[s] for s in folds}
        helps = all(g >= MIN_GAIN for g in gains.values())
        keep = helps and name not in LEAKY
        if keep:
            kept.append(name)
        rows.append({
            "variable": LABELS[name], "cle": name, "renseignee": round(float(values.notna().mean()), 3),
            "correlation_residu": round(float(pd.Series(resid).corr(values.reset_index(drop=True))), 3),
            **{f"gain_{s}": round(100 * g, 2) for s, g in gains.items()},
            "recommandation": "à activer" if keep else (
                "gain suspect : fuite probable" if helps and name in LEAKY else "inutile"),
        })
    table = pd.DataFrame(rows)
    combined = None
    if len(kept) > 1:
        combined = {s: round(100 * (1 - _oof_error(df, f, tuple(kept), ridge) / base[s]), 2) for s, f in folds.items()}

    # Redondance duration / maturité.
    redundancy = None
    if "duration" in df.columns and pd.to_numeric(df["duration"], errors="coerce").notna().mean() > 0.5:
        redundancy = round(float(pd.to_numeric(df["duration"], errors="coerce").corr(df["log_t"])), 3)

    # Prix et rendement : à quel point devinent-ils le spread ? (contrôle, pas variable)
    leakage = []
    for column, label in (("ytm", "Rendement (YTM)"), ("price", "Prix")):
        if column in df.columns and pd.to_numeric(df[column], errors="coerce").notna().mean() > 0.5:
            values = pd.to_numeric(df[column], errors="coerce")
            ok = values.notna().to_numpy()
            share = 0.0
            for currency in df["currency"].unique():
                mask = ok & (df["currency"].to_numpy() == currency)
                if mask.sum() > 20:
                    fit = np.polyfit(values[mask], y[mask], 2)
                    share += ((y[mask] - np.polyval(fit, values[mask])) ** 2).sum()
            total = ((y[ok] - y[ok].mean()) ** 2).sum()
            leakage.append({"colonne": label, "part_du_spread_devinee": round(float(1 - share / total), 3)})

    findings = [f"Courbe des pairs sans variable supplémentaire : erreur médiane hors échantillon de "
                f"{100 * base['émetteur connu']:.1f} % (émetteur connu) et {100 * base['émetteur inconnu']:.1f} % "
                "(émetteur inconnu)."]
    for row in rows:
        findings.append(
            f"{row['variable']} : renseignée à {100 * row['renseignee']:.0f} % ; erreur "
            f"{-row['gain_émetteur connu']:+.1f} % (émetteur connu), {-row['gain_émetteur inconnu']:+.1f} % "
            f"(émetteur inconnu) → {row['recommandation']}."
        )
    if combined:
        findings.append(f"Variables retenues ensemble : erreur {-combined['émetteur connu']:+.1f} % et "
                        f"{-combined['émetteur inconnu']:+.1f} %.")
    if redundancy is not None:
        findings.append(f"Corrélation entre duration et log de la maturité : {redundancy:.2f}"
                        + (" ; la duration répète presque la maturité." if abs(redundancy) > 0.9 else "."))
    if "duration" in available:
        findings.append(
            "Attention : une duration modifiée se calcule au rendement de marché, qui contient le spread. Ce qu'elle "
            "apporte au-delà de la maturité peut venir du spread lui-même (fuite). À n'activer que si elle est calculée "
            "sans le rendement de l'obligation (taux de référence), ou si son gain persiste en la recalculant ainsi.")
    for item in leakage:
        share = item["part_du_spread_devinee"]
        detail = (f"devine {100 * share:.0f} % de la dispersion des spreads à lui seul" if share >= 0.2 else
                  "seul, il devine peu (il dépend aussi du coupon et de la maturité), mais avec eux il redonne "
                  "le rendement, donc le spread")
        findings.append(f"{item['colonne']} : {detail}. Lié mécaniquement au spread : à ne jamais utiliser comme "
                        "variable, seulement pour des contrôles.")
    if not available:
        findings.append("Aucune variable supplémentaire (ESG, date d'émission, duration) n'a été trouvée dans la base.")
    findings.append("Réglage à reporter dans config/settings.yaml, section peer_curve : extra: "
                    + (str(kept) if kept else "[]") + ".")
    return {"title": "Colonnes supplémentaires", "findings": findings, "recommended": kept,
            "tables": {"variables": table, "fuite": pd.DataFrame(leakage), "combinees": combined,
                       "base": base}}
