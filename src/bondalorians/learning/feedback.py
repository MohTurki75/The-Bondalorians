"""Retours utilisateurs transformés en observations pondérées.

Un retour devient une pseudo-obligation ajoutée à l'univers : elle agit comme
un comparable supplémentaire et pèse dans la courbe des pairs. Un prix traité
pèse plus qu'un avis d'expert, qui n'est pas une vérité terrain.
Aucun modèle n'est promu automatiquement : le réentraînement complet reste un
script, validé par le backtest.
"""

from __future__ import annotations

import pandas as pd

from bondalorians.data.schema import DESCRIPTIVE_COLUMNS

KINDS = ("trade", "opinion")


def validate_feedback(kind: str, z_spread_bp: float) -> None:
    if kind not in KINDS:
        raise ValueError(f"Type de retour inconnu : {kind!r} (attendu : {KINDS}).")
    if not z_spread_bp or z_spread_bp <= 0:
        raise ValueError("Le spread corrigé doit être strictement positif.")


def feedback_to_observations(feedback_records: list[dict], settings: dict) -> pd.DataFrame:
    """Construit les pseudo-obligations à partir des retours enregistrés."""
    weights = settings["feedback"]["weights"]
    rows = []
    for record in feedback_records:
        payload = record["payload"]
        row = {c: payload["bond"].get(c) for c in DESCRIPTIVE_COLUMNS}
        row["isin"] = f"FB_{record['id']}"
        row["z_spread"] = float(payload["z_spread_bp"])
        row["obs_weight"] = float(weights[payload["kind"]])
        rows.append(row)
    return pd.DataFrame(rows)


def augment_universe(universe: pd.DataFrame, feedback_records: list[dict], settings: dict) -> pd.DataFrame:
    if not feedback_records:
        return universe
    base = universe.copy()
    if "obs_weight" not in base.columns:
        base["obs_weight"] = 1.0
    return pd.concat([base, feedback_to_observations(feedback_records, settings)], ignore_index=True)
