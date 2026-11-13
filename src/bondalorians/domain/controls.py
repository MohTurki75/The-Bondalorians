"""Contrôles de cohérence appliqués à une estimation."""

from __future__ import annotations

OK, WARNING, FAILED = "ok", "alerte", "echec"


def run_controls(
    estimate_bp: float,
    segment_bounds_bp: tuple[float, float] | None,
    senior_reference_bp: float | None,
    model_gap_log: float | None,
    n_eff: float,
    issuer_lambda: float,
    settings: dict,
) -> list[dict]:
    cfg = settings["controls"]
    checks = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"control": name, "status": status, "detail": detail})

    add("Spread positif", OK if estimate_bp > 0 else FAILED, f"{estimate_bp:.1f} pb")

    if segment_bounds_bp is None:
        add("Plage du segment de notation", WARNING, "Aucune obligation de notation voisine dans l'univers.")
    else:
        low, high = segment_bounds_bp
        inside = low <= estimate_bp <= high
        add(
            "Plage du segment de notation",
            OK if inside else WARNING,
            f"Estimation {estimate_bp:.0f} pb ; plage 1 %-99 % des notations voisines : {low:.0f} à {high:.0f} pb.",
        )

    if senior_reference_bp is not None:
        coherent = estimate_bp >= senior_reference_bp
        add(
            "Cohérence de subordination",
            OK if coherent else WARNING,
            f"Dette plus senior du même émetteur : médiane {senior_reference_bp:.0f} pb.",
        )

    if model_gap_log is not None:
        within = abs(model_gap_log) <= cfg["max_model_gap"]
        add(
            "Accord champion-challenger",
            OK if within else WARNING,
            f"Écart relatif entre les deux modèles : {100 * (abs(model_gap_log)):.0f} % (en log).",
        )

    enough = n_eff >= cfg["min_effective_comparables"] or issuer_lambda >= 0.5
    add(
        "Comparables suffisants",
        OK if enough else WARNING,
        f"Nombre effectif de comparables hors émetteur : {n_eff:.1f} ; poids de la courbe émetteur : {issuer_lambda:.2f}.",
    )
    return checks
