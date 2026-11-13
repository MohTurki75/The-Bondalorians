"""Score de confiance sur 100, à partir de quatre composantes mesurables.

Les poids sont des choix de départ (config/settings.yaml). Le test de
crédibilité est dans le backtest : l'erreur réalisée doit décroître quand le
score augmente.
"""

from __future__ import annotations

import numpy as np


def _clip(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def confidence_score(
    half_width_log: float | None,
    n_eff: float,
    mean_distance: float,
    issuer_lambda: float,
    model_gap_log: float | None,
    segment_count: int,
    settings: dict,
) -> dict:
    cfg = settings["confidence"]
    components: dict[str, float] = {}
    if half_width_log is not None:
        components["interval"] = _clip(1 - half_width_log / cfg["interval_ref"])
    peers_quality = _clip(n_eff / 8.0) * _clip(1 - mean_distance / 0.5)
    components["comparables"] = max(_clip(issuer_lambda), peers_quality)
    if model_gap_log is not None:
        components["agreement"] = _clip(1 - abs(model_gap_log) / cfg["agreement_ref"])
    components["domain"] = _clip(segment_count / cfg["domain_ref"])

    # Une composante indisponible est retirée et les poids sont renormalisés.
    weights = {k: cfg["weights"][k] for k in components}
    total = sum(weights.values())
    score = 100 * sum(components[k] * weights[k] for k in components) / total
    grades = cfg["grades"]
    grade = "A" if score >= grades["A"] else "B" if score >= grades["B"] else "C"
    return {
        "score": round(score, 1),
        "grade": grade,
        "components": {k: round(v, 3) for k, v in components.items()},
    }
