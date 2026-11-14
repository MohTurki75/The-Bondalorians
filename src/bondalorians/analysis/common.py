"""Outils partagés par les analyses."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from bondalorians.data.schema import RATING_SCALE

TIER_LABELS = {
    "simple": "Simple", "intermediaire": "Intermédiaire", "complexe": "Complexe",
    "emetteur_inconnu": "Émetteur inconnu",
}


def real_bonds(estimator) -> pd.DataFrame:
    """Univers sans les pseudo-obligations issues des retours utilisateurs."""
    df = estimator.df_
    return df[~df["isin"].astype(str).str.startswith("FB_")].reset_index(drop=True)


def notch_label(notch: float) -> str:
    index = int(round(notch)) - 1
    return RATING_SCALE[min(max(index, 0), len(RATING_SCALE) - 1)]


def rating_bucket(notch: float) -> str:
    """Regroupe les crans en grandes lettres : AAA, AA, A, BBB, BB, B, CCC et moins."""
    for limit, label in ((1, "AAA"), (4, "AA"), (7, "A"), (10, "BBB"), (13, "BB"), (16, "B")):
        if notch <= limit:
            return label
    return "CCC et moins"


BUCKET_ORDER = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC et moins"]


def fmt(value: float, digits: int = 0) -> str:
    """Nombre au format français : 1 234,5."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    text = f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")
    return text


def pct(value: float, digits: int = 0) -> str:
    return f"{fmt(100 * value, digits)} %"


def jsonable(value):
    if isinstance(value, pd.DataFrame):
        return jsonable(value.to_dict("records"))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def for_agent(analysis: dict) -> dict:
    """Version sans les données détaillées, pour l'agent et le serveur MCP."""
    return jsonable({k: v for k, v in analysis.items() if k != "points"})


def r2(y: np.ndarray, prediction: np.ndarray) -> float:
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    total = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - prediction) ** 2).sum() / total) if total > 0 else float("nan")
