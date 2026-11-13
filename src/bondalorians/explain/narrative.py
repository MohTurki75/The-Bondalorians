"""Récit déterministe d'une estimation.

Sert de texte de secours sans LLM et de source unique de chiffres pour le LLM :
celui-ci reformule ce texte et le JSON, il n'introduit aucun nombre.
"""

from __future__ import annotations

MODEL_NAMES = {"A": "comparables ajustés par la courbe des pairs", "B": "CatBoost (challenger ML)"}

STANDING_CAVEATS = [
    "L'estimation repose uniquement sur les données de la base : liquidité réelle, "
    "perspective de notation et actualité de l'émetteur ne sont pas observées.",
    "Aucune prime de nouvelle émission n'est ajoutée.",
]


def build_narrative(result: dict) -> str:
    bond = result["bond"]
    lines = [
        f"Z-Spread juste estimé : {result['z_spread_bp']:.0f} pb pour {bond['issuer']} "
        f"({bond['rating']}, {bond['seniority']}, {bond['maturity_years']:.1f} ans, {bond['currency']})."
    ]
    if result.get("interval_bp"):
        low, high = result["interval_bp"]
        lines.append(
            f"Intervalle à {100 * result['interval_level']:.0f} % : {low:.0f} à {high:.0f} pb."
        )
    conf = result["confidence"]
    lines.append(f"Confiance : {conf['score']:.0f}/100 (note {conf['grade']}).")
    lines.append(f"Palier : {result['tier']}. {result['tier_reason']}")
    lines.append(f"Modèle retenu : {MODEL_NAMES[result['champion']]}.")

    offset = result["issuer_offset"]
    if offset["n_bonds"] > 0:
        lines.append(
            f"Courbe émetteur : {offset['n_bonds']} obligation(s), écart moyen aux pairs "
            f"{100 * offset['mean_residual_log']:+.0f} %, retenu à {100 * offset['lambda']:.0f} %."
        )
    lines.append(
        f"Comparables hors émetteur : {len(result['comparables'])} retenus "
        f"(filtre : {result['filter_level']})."
    )
    drivers = sorted(
        (s for s in result["waterfall"] if s["delta_bp"] is not None),
        key=lambda s: -abs(s["delta_bp"]),
    )[:3]
    lines.append(
        "Principaux écarts à l'obligation moyenne : "
        + " ; ".join(f"{s['step']} {s['delta_bp']:+.0f} pb" for s in drivers) + "."
    )
    alerts = [c for c in result["controls"] if c["status"] != "ok"]
    if alerts:
        lines.append("Contrôles en alerte : " + " ; ".join(f"{c['control']} ({c['detail']})" for c in alerts))
    else:
        lines.append("Tous les contrôles de cohérence sont passés.")
    lines.append("Réserves : " + " ".join(result["caveats"]))
    return "\n".join(lines)
