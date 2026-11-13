"""Évaluation sur un échantillon mis de côté : erreur et couverture de l'intervalle.

Contrairement au backtest interne (qui sert à calibrer), cet échantillon n'a
servi ni à l'apprentissage ni à la calibration : la couverture mesurée est honnête.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bondalorians.data.schema import prepare
from bondalorians.quant.estimator import ZSpreadEstimator


def holdout_evaluation(
    universe: pd.DataFrame,
    settings: dict | None = None,
    use_ml: bool = False,
    by_issuer: bool = False,
    test_share: float = 0.2,
    seed: int = 11,
) -> pd.DataFrame:
    """Met de côté des obligations (ou des émetteurs entiers) et les estime.

    Renvoie une ligne par obligation testée : vérité, estimation, bornes, palier, score.
    """
    df = prepare(universe, require_target=True)
    rng = np.random.default_rng(seed)
    if by_issuer:
        issuers = df["issuer"].unique()
        held = set(rng.choice(issuers, size=max(1, int(test_share * len(issuers))), replace=False))
        is_test = df["issuer"].isin(held).to_numpy()
    else:
        is_test = rng.random(len(df)) < test_share

    estimator = ZSpreadEstimator(settings, use_ml=use_ml).fit(df[~is_test], calibrate=True)
    rows = []
    for bond in df[is_test].to_dict("records"):
        truth = bond.pop("z_spread")
        result = estimator.estimate(bond)
        interval = result["interval_bp"]
        rows.append({
            "truth_bp": truth,
            "estimate_bp": result["z_spread_bp"],
            "low_bp": interval[0] if interval else np.nan,
            "high_bp": interval[1] if interval else np.nan,
            "tier": result["tier"] if result["issuer_seen"] else "emetteur_inconnu",
            "score": result["confidence"]["score"],
        })
    out = pd.DataFrame(rows)
    out["abs_error_bp"] = (out["estimate_bp"] - out["truth_bp"]).abs()
    out["covered"] = (out["truth_bp"] >= out["low_bp"]) & (out["truth_bp"] <= out["high_bp"])
    out["relative_width"] = (out["high_bp"] - out["low_bp"]) / out["estimate_bp"]
    out.attrs["calibration_report"] = estimator.calibration_report()
    return out


def summarize(evaluation: pd.DataFrame) -> pd.DataFrame:
    with_interval = evaluation[evaluation["low_bp"].notna()]
    table = evaluation.groupby("tier").agg(
        n=("truth_bp", "size"),
        mae_bp=("abs_error_bp", "mean"),
        mediane_abs_bp=("abs_error_bp", "median"),
    )
    table["couverture"] = with_interval.groupby("tier")["covered"].mean()
    table["largeur_relative"] = with_interval.groupby("tier")["relative_width"].mean()
    return table.round(3).reset_index()
