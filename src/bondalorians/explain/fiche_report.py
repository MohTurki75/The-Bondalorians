"""Explication d'une estimation faite depuis une fiche obligataire, en Markdown.

Tous les chiffres viennent du résultat de l'estimateur ; ce module ne fait que les mettre en forme.
"""

from __future__ import annotations

from bondalorians.explain.narrative import MODEL_NAMES

MODEL_LABELS = {
    "courbe_pairs_bp": "P · courbe des pairs", "A_comparables_bp": "A · comparables ajustés",
    "B_catboost_bp": "B · CatBoost indépendant", "C_comparables_corriges_bp": "C · comparables corrigés",
    "S_ancrage_senior_bp": "S · ancrage sur la dette senior", "top_down_bp": "lecture descendante (contrôle)",
}
STATUS = {"ok": "✓", "alerte": "⚠", "echec": "✕"}


def _fmt(value, digits: int = 0, unit: str = "") -> str:
    if value is None:
        return "—"
    text = f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")
    return f"{text}{unit}"


def _day(text) -> str:
    """« Fri Aug 17 2035 02:00:00 GMT+0200 (…) » -> « 17/08/2035 »."""
    import pandas as pd

    from bondalorians.data.repository import _to_dates

    if not text:
        return "—"
    parsed = _to_dates(pd.Series([str(text)])).iloc[0]
    return str(text) if pd.isna(parsed) else parsed.strftime("%d/%m/%Y")


def fiche_markdown(raw: dict, bond: dict, notes: list[str], result: dict, sensitivity: list[dict] | None,
                   observed: dict | None) -> str:
    low, high = result["interval_bp"] or (None, None)
    conf = result["confidence"]
    lines = [
        f"# {raw.get('id', '?')} · {raw.get('nomobligation', '')}",
        "",
        f"**Z-Spread juste estimé : {_fmt(result['z_spread_bp'])} pb** "
        f"(intervalle à {100 * result['interval_level']:.0f} % : {_fmt(low)} à {_fmt(high)} pb ; "
        f"confiance {_fmt(conf.get('score'))}/100, note {conf.get('grade', '—')}).",
        "",
        f"Modèle retenu : {MODEL_NAMES.get(result['champion'], result['champion'])}. "
        f"Palier : {result['tier']} ({result['tier_reason']}).",
        "",
        "## Ce qui a été lu",
        "",
        "| Caractéristique | Fiche | Retenu pour l'estimation |",
        "|---|---|---|",
        f"| Émetteur | {raw.get('ticker')} | {bond['issuer']} ({'connu' if result['issuer_seen'] else 'absent de la base'}) |",
        f"| Notation | {raw.get('ratingmoyen')} | {bond['rating']} ({bond.get('rating_source', 'lue sur la fiche')}) |",
        f"| Séniorité | {raw.get('rangseniorit_obligation')} | {bond['seniority']} |",
        f"| Secteur / industrie | {raw.get('nomsecteur')} / {raw.get('industrieemetteur')} | {bond['sector']} |",
        f"| Pays, devise | {raw.get('paysemetteur')}, {raw.get('devise')} | {bond['country']}, {bond['currency']} |",
        f"| Maturité de calcul | {_day(raw.get('datematurit_pricingobligation'))} | {_fmt(bond['maturity_years'], 2)} ans |",
        f"| Échéance finale | {_day(raw.get('datematurit'))} | {_fmt(bond.get('final_maturity_years'), 2)} ans |",
        f"| Remboursement anticipé | {raw.get('typematurit')} | {'oui' if bond.get('callable') else 'non'} |",
        f"| Coupon | {raw.get('typecoupon')} | {bond['coupon_type']} |",
        f"| Structure | {raw.get('structureaffaire')} | {bond.get('structure')} |",
        f"| Verte, bail-in | {raw.get('greenbond')}, {raw.get('bondbailin')} | "
        f"{'oui' if bond.get('green') else 'non'}, {bond.get('bail_in')} |",
        "",
    ]
    if notes:
        lines += ["## Points d'attention", ""] + [f"- {n}" for n in notes] + [""]
    if observed:
        lines += ["## Obligation déjà dans la base", "",
                  f"Z-Spread observé dans la base : {_fmt(observed['z_spread_bp'])} pb. Il n'a pas servi à "
                  "l'estimation (l'obligation est exclue de ses propres comparables) ; écart estimation − observé : "
                  f"{_fmt(result['z_spread_bp'] - observed['z_spread_bp'], 0)} pb.", ""]

    lines += ["## Les modèles", "", "| Modèle | Z-Spread (pb) |", "|---|---|"]
    for key, value in result["models"].items():
        if value is not None:
            lines.append(f"| {MODEL_LABELS.get(key, key)} | {_fmt(value)} |")
    top_down = result.get("top_down")
    if top_down and top_down.get("z_spread_bp"):
        lines += ["", f"Lecture de contrôle ({top_down.get('reader', '')}) : {_fmt(top_down['z_spread_bp'])} pb, "
                  f"{top_down.get('level_name', top_down.get('method', ''))}."]

    lines += ["", "## Du spread moyen à l'estimation", "", "| Étape | Effet (pb) | Cumul (pb) |", "|---|---|---|"]
    for step in result["waterfall"]:
        if step["delta_bp"] == 0:
            continue  # variable inactive ou sans effet
        delta = "" if step["delta_bp"] is None else f"{step['delta_bp']:+.1f}".replace(".", ",")
        lines.append(f"| {step['step']} | {delta} | {_fmt(step['cumulative_bp'])} |")

    lines += ["", "## Comparables les plus proches", "",
              "| Émetteur | Notation | Séniorité | Secteur | Maturité | Z-Spread | Ajusté | Poids |",
              "|---|---|---|---|---|---|---|---|"]
    for c in result["comparables"][:8]:
        lines.append(f"| {c['issuer']} | {c['rating']} | {c['seniority']} | {c['sector']} | "
                     f"{_fmt(c['maturity_years'], 1)} | {_fmt(c['z_spread_bp'])} | {_fmt(c['adjusted_z_spread_bp'])} | "
                     f"{_fmt(c['weight'], 2)} |")

    if sensitivity:
        lines += ["", "## Sensibilité à la notation (notation déduite)", "",
                  "| Notation | Z-Spread (pb) |", "|---|---|"]
        lines += [f"| {s['rating']}{' (retenue)' if s['retenue'] else ''} | {_fmt(s['z_spread_bp'])} |"
                  for s in sensitivity]

    lines += ["", "## Contrôles", ""]
    lines += [f"- {STATUS.get(c['status'], c['status'])} {c['control']} : {c['detail']}" for c in result["controls"]]
    lines += ["", "## Réserves", ""] + [f"- {c}" for c in result["caveats"]]
    lines += ["", f"_Version du modèle {result['model_version']} · estimation {result.get('run_id', '')}_", ""]
    return "\n".join(lines)
