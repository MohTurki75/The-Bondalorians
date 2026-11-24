"""Cascade en points de base : de l'obligation moyenne de l'univers à l'estimation."""

from __future__ import annotations

import math

from bondalorians.quant.peer_curve import GROUP_LABELS


def build_waterfall(
    mean_log: float,
    group_contributions: dict[str, float],
    issuer_term_log: float,
    issuer_label: str | None,
    peers_term_log: float,
    final_log: float,
    peers_label: str | None = "Comparables hors émetteur",
) -> list[dict]:
    """Chaque étape est un écart en log converti en points de base, dans l'ordre.

    La décomposition est séquentielle : le montant en pb d'une étape dépend des
    étapes précédentes (modèle multiplicatif). La somme est exacte.
    """
    steps = [{"step": "Obligation moyenne de l'univers", "delta_bp": None, "cumulative_bp": math.exp(mean_log)}]
    cumulative = mean_log

    def add(label: str, delta_log: float) -> None:
        nonlocal cumulative
        before = math.exp(cumulative)
        cumulative += delta_log
        after = math.exp(cumulative)
        steps.append({"step": label, "delta_bp": after - before, "cumulative_bp": after})

    for group, label in GROUP_LABELS.items():
        if group in group_contributions:  # variable absente du modèle : pas de ligne à zéro
            add(label, group_contributions[group])
    # Une étape sans libellé (sans objet : émetteur inconnu, courbe seule retenue) est omise.
    if issuer_label is not None:
        add(issuer_label, issuer_term_log)
    if peers_label is not None:
        add(peers_label, peers_term_log)
    residual = final_log - cumulative
    if abs(residual) > 1e-9:
        add("Ajustement du modèle retenu", residual)
    for step in steps:
        step["cumulative_bp"] = round(step["cumulative_bp"], 1)
        if step["delta_bp"] is not None:
            step["delta_bp"] = round(step["delta_bp"], 1)
    return steps
