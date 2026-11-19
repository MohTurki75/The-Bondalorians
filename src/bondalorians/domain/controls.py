"""Contrôles de cohérence appliqués à une estimation."""

from __future__ import annotations

OK, WARNING, FAILED = "ok", "alerte", "echec"


def fr(value: float, digits: int = 0) -> str:
    """Nombre au format français (virgule décimale, espace des milliers)."""
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def run_controls(
    estimate_bp: float,
    segment_bounds_bp: tuple[float, float] | None,
    senior_check: dict | None,
    model_gap_log: float | None,
    n_eff: float,
    issuer_lambda: float,
    settings: dict,
) -> list[dict]:
    cfg = settings["controls"]
    checks = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"control": name, "status": status, "detail": detail})

    add("Spread positif", OK if estimate_bp > 0 else FAILED, f"{fr(estimate_bp, 1)} pb")

    if segment_bounds_bp is None:
        add("Plage du segment de notation", WARNING, "Aucune obligation de notation voisine dans l'univers.")
    else:
        low, high = segment_bounds_bp
        inside = low <= estimate_bp <= high
        add(
            "Plage du segment de notation",
            OK if inside else WARNING,
            f"Estimation {fr(estimate_bp)} pb ; plage 1 %-99 % des notations voisines : {fr(low)} à {fr(high)} pb.",
        )

    if senior_check is not None:
        # La dette senior de l'émetteur est ramenée au profil de la cible (maturité, notation,
        # devise…) avant comparaison : une médiane brute mélangerait des maturités différentes.
        equivalent = senior_check["equivalent_bp"]
        coherent = estimate_bp >= equivalent
        add(
            "Cohérence de subordination",
            OK if coherent else WARNING,
            f"Dette senior du même émetteur ({senior_check['n']} obligation(s)) ramenée au profil de la cible : "
            f"{fr(equivalent)} pb ; médiane brute {fr(senior_check['raw_median_bp'])} pb. "
            + ("L'estimation est au-dessus." if coherent else "L'estimation est en dessous : à examiner."),
        )

    if model_gap_log is not None:
        within = abs(model_gap_log) <= cfg["max_model_gap"]
        add(
            "Accord champion-challenger",
            OK if within else WARNING,
            f"Écart relatif entre les deux modèles : {fr(100 * abs(model_gap_log))} % (en log).",
        )

    enough = n_eff >= cfg["min_effective_comparables"] or issuer_lambda >= 0.5
    add(
        "Comparables suffisants",
        OK if enough else WARNING,
        f"Nombre effectif de comparables hors émetteur : {fr(n_eff, 1)} ; poids de la courbe émetteur : "
        f"{fr(100 * issuer_lambda)} %.",
    )
    return checks
