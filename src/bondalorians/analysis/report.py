"""Rapport Markdown des analyses, à donner à un LLM pour interprétation.

Le rapport contient une consigne, un glossaire, tous les constats calculés et
les tableaux résumés. Aucun chiffre n'y est écrit à la main : tout vient des
fonctions d'analyse.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from bondalorians.analysis.clustering import METHODS, cluster_universe
from bondalorians.analysis.common import TIER_LABELS, fmt, real_bonds
from bondalorians.analysis.models import all_model_views, comparables_view
from bondalorians.analysis.sample import describe_sample

MODEL_LABELS = {
    "courbe_pairs_bp": "courbe des pairs", "A_comparables_bp": "A", "B_catboost_bp": "B (CatBoost)",
    "C_comparables_corriges_bp": "C (comparables corrigés)", "top_down_bp": "lecture descendante",
}
CHAMPION_LABELS = {"A": "A (comparables)", "B": "B (CatBoost)", "C": "C (comparables corrigés)",
                   "P": "courbe des pairs seule"}

INSTRUCTIONS = """Tu es analyste crédit et data scientist. Tu expliques ces résultats à un gérant de portefeuille
qui n'est pas spécialiste des données. Le système estime le Z-Spread juste d'une obligation à partir d'un
univers d'obligations comparables ; ce document décrit l'échantillon, chaque modèle et un clustering.

Règles :
1. N'utilise que les chiffres de ce document. N'en invente aucun ; si un calcul te semble utile, dis-le
   sans le faire.
2. Distingue ce qui est mesuré ici (fait), ce que tu en déduis (interprétation) et ce que tu supposes
   (hypothèse).
3. Toutes les erreurs citées sont hors échantillon : chaque obligation a été estimée par un modèle qui ne
   l'avait pas vue.
4. Si la source est « données synthétiques », rappelle que ces chiffres valident la mécanique et ne disent
   rien du marché réel.
5. Sois critique : signale ce qui est fragile, contradictoire ou insuffisamment démontré.

Réponds, dans cet ordre :
1. Que dit l'échantillon, et qu'est-ce qui détermine le niveau des spreads ?
2. Quel modèle est le plus fiable, dans quels cas, et l'apprentissage automatique apporte-t-il quelque chose ?
3. Le score de confiance et l'intervalle à 90 % sont-ils dignes de confiance ? Que disent les tests de
   la section 6 (biais, stabilité, apport, surapprentissage, couverture) ?
4. Le clustering révèle-t-il une vraie structure, ou seulement un découpage de lecture ?
5. Si une obligation est décrite en fin de document : comment se situe-t-elle, et avec quelle confiance ?
6. Les trois faiblesses principales du système, et ce qu'il faudrait vérifier sur les données réelles."""

GLOSSARY = """| Terme | Sens |
|---|---|
| pb | Point de base : 0,01 %. |
| Médiane | Valeur du milieu : la moitié des cas est au-dessous. |
| Hors échantillon | Estimé par un modèle qui n'a pas vu l'obligation. |
| Part de la dispersion expliquée | De 0 à 100 % : à quel point une information permet de deviner le spread (en logarithme). |
| Courbe des pairs | Régression du log du spread sur notation, maturité, séniorité, secteur, devise. |
| Modèle A | Courbe des pairs + écart propre à l'émetteur (rétréci s'il a peu d'obligations) + écart des comparables. |
| Modèle B | CatBoost (arbres de décision en série) sans l'émetteur comme variable. |
| Modèle C | Modèle A corrigé par un CatBoost qui apprend ses erreurs. |
| Forêt aléatoire | Moyenne de centaines d'arbres de décision ; modèle de comparaison, hors estimation. |
| k | Dans le modèle A, l'écart d'un émetteur à n obligations est retenu à n / (n + k). |
| SHAP | Contribution de chaque variable à une prédiction. |
| Couverture | Part des cas où le vrai spread tombe dans l'intervalle annoncé. |
| Silhouette | De −1 à 1 : séparation des familles d'un clustering ; sous 0,25, elles se chevauchent. |
| Indice de Rand ajusté | De 0 (hasard) à 1 (identique) : accord entre deux clusterings. |
| Palier | Simple, intermédiaire, complexe ou émetteur inconnu, selon l'information disponible sur l'émetteur. |"""


def md_table(frame: pd.DataFrame, columns: dict[str, str] | None = None, digits: int = 1) -> str:
    """Tableau Markdown, nombres au format français."""
    if frame is None or len(frame) == 0:
        return "_Aucune donnée._"
    frame = frame.copy()
    if columns:
        frame = frame[[c for c in columns if c in frame.columns]].rename(columns=columns)

    def cell(value) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "—"
        if isinstance(value, bool):
            return "oui" if value else "non"
        if isinstance(value, int):
            return fmt(value)
        if isinstance(value, float):
            return fmt(value, 0 if abs(value) >= 100 else digits)
        return str(value).replace("|", "/")

    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def _findings(analysis: dict) -> str:
    return "\n".join(f"- {f}" for f in analysis["findings"])


def build_report(estimator, result: dict | None = None, source: str = "",
                 include_forest: bool = True, methods=tuple(METHODS), include_quality: bool = True,
                 quality_cases: int | None = 1500) -> str:
    """Rapport complet. `result` : estimation à situer (facultatif)."""
    if estimator.calibration_ is None:
        raise RuntimeError("Estimateur non calibré : le rapport a besoin du backtest.")
    sample = describe_sample(estimator)
    views = all_model_views(estimator, include_forest=include_forest)
    clusters = {m: cluster_universe(estimator, m, target=result) for m in methods}
    n = len(real_bonds(estimator))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    out = [
        "# Analyses de l'échantillon et des modèles",
        "",
        f"- Généré le : {now}",
        f"- Source : {source or 'non précisée'}",
        f"- Obligations : {fmt(n)} ; version du modèle : `{estimator.model_version_}`",
        f"- CatBoost : {'actif' if estimator.ml_ is not None else 'inactif'} ; "
        f"forêt aléatoire : {'incluse' if include_forest else 'non calculée'}",
        "",
        "## Consigne",
        "",
        INSTRUCTIONS,
        "",
        "## Glossaire",
        "",
        GLOSSARY,
        "",
        "## 1. Échantillon",
        "",
        _findings(sample),
        "",
        "### Spread par grande notation (pb)",
        "",
        md_table(sample["tables"]["spread_par_lettre"], {
            "lettre": "Notation", "n": "Obligations", "p10": "10e centile", "p25": "25e centile",
            "mediane": "Médiane", "p75": "75e centile", "p90": "90e centile"}),
        "",
        "### Part de la dispersion des spreads expliquée par chaque caractéristique seule",
        "",
        md_table(sample["tables"]["part_expliquee"].assign(part_expliquee=lambda d: 100 * d["part_expliquee"]),
                 {"caracteristique": "Caractéristique", "part_expliquee": "Part expliquée (%)"}),
        "",
        "## 2. Les modèles un par un",
        "",
    ]
    sections = [("courbe", None), ("emetteur", None), ("reglage", None), ("catboost", "importance"),
                ("correcteur", None), ("foret", "importance"), ("effets", None)]
    for key, table in sections:
        if key not in views:
            continue
        view = views[key]
        out += [f"### {view['title']}", "", _findings(view), ""]
        if table and view.get("tables", {}).get(table) is not None:
            frame = view["tables"][table]
            columns = ({"variable": "Variable", "effet_moyen_pct": "Effet moyen sur le spread (%)"} if key == "catboost"
                       else {"variable": "Variable", "hausse_erreur_pct": "Hausse de l'erreur si brouillée (points de %)"})
            out += [md_table(frame, columns), ""]
        if key == "reglage":
            params = estimator.params_
            out += [md_table(pd.DataFrame([{"Critère": k, "Poids": v} for k, v in params["weights"].items()]), digits=3),
                    "", f"Nombre maximal de comparables : {params['top_k']} ; largeur du noyau : "
                        f"{fmt(params['bandwidth'], 2)}.", ""]

    comparison = views["comparaison"]
    pivot = comparison["tables"]["erreurs"].pivot(index="palier", columns="modele", values="erreur_mediane_bp")
    counts = comparison["tables"]["erreurs"].groupby("palier")["n"].first()
    pivot.insert(0, "Obligations testées", counts)
    confidence = views["confiance"]
    out += [
        "## 3. Comparaison des modèles et confiance",
        "",
        "### Erreur médiane hors échantillon (pb), par palier et par modèle",
        "",
        md_table(pivot.reset_index().rename(columns={"palier": "Palier"})),
        "",
        _findings(comparison),
        "",
        "### Score de confiance : erreur par tranche de difficulté prévue",
        "",
        md_table(confidence["tables"]["par_decile"], {
            "decile": "Tranche (1 = plus facile)", "erreur_prevue_pct": "Erreur prévue (%)",
            "erreur_mediane_pct": "Erreur réalisée (%)"}),
        "",
        "### Intervalle à 90 % : couverture par palier",
        "",
        md_table(confidence["tables"]["couverture"].assign(couverture=lambda d: 100 * d["couverture"]), {
            "palier": "Palier", "n": "Obligations", "couverture": "Couverture (%)",
            "largeur_mediane_pct": "Largeur médiane (%)"}),
        "",
        _findings(confidence),
        "",
        "## 4. Clustering",
        "",
        "Le clustering regroupe les obligations par leurs caractéristiques (notation, maturité, séniorité, "
        "taille, secteur, devise), sans le spread. Il n'intervient pas dans l'estimation.",
        "",
    ]
    first = next(iter(clusters.values()))
    out += [
        "### Combien de familles ? (silhouette du k-means)",
        "",
        md_table(first["tables"]["silhouette"], {"k": "Familles", "silhouette": "Silhouette"}, digits=3),
        "",
        f"- {first['tables']['structure']['reason']}",
        "",
        "### Accord entre méthodes (indice de Rand ajusté)",
        "",
        md_table(first["tables"]["accord"].rename(columns={"methode": "Méthode"}), digits=2),
        "",
    ]
    for method, analysis in clusters.items():
        out += [
            f"### {analysis['title']}",
            "",
            _findings(analysis),
            "",
            md_table(analysis["tables"]["familles"], {
                "famille": "Famille", "nom": "Description", "obligations": "Obligations", "emetteurs": "Émetteurs",
                "notation_mediane": "Notation médiane", "maturite_mediane": "Maturité médiane (ans)",
                "secteur_principal": "Secteur principal", "spread_p25": "Spread 25e centile",
                "spread_median": "Spread médian", "spread_p75": "Spread 75e centile"}),
            "",
        ]

    if result is not None:
        bond = result["bond"]
        conf = result["confidence"]
        comps = comparables_view(estimator, result)
        out += [
            "## 5. Obligation estimée",
            "",
            md_table(pd.DataFrame([{
                "Émetteur": bond["issuer"], "Notation": bond["rating"], "Séniorité": bond["seniority"],
                "Maturité (ans)": bond["maturity_years"], "Devise": bond["currency"], "Secteur": bond["sector"]}])),
            "",
            f"- Z-Spread juste estimé : {fmt(result['z_spread_bp'])} pb ; intervalle à "
            f"{fmt(100 * result['interval_level'])} % : "
            + (f"{fmt(result['interval_bp'][0])} à {fmt(result['interval_bp'][1])} pb." if result["interval_bp"]
               else "indisponible."),
            f"- Confiance : {fmt(conf['score'])}/100 (note {conf['grade']}) ; erreur typique attendue "
            f"{fmt(conf['typical_error_bp'])} pb." if conf.get("score") is not None else "- Confiance : indisponible.",
            (f"- Spread observé dans la base pour cette obligation : {fmt(result['observed_z_spread_bp'])} pb. "
             "Il n'a pas servi à l'estimation (l'obligation est exclue de ses propres comparables)."
             if result.get("observed_z_spread_bp") else "- Obligation absente de la base : aucun spread observé."),
            f"- Palier : {TIER_LABELS[result['tier']]}. {result['tier_reason']}",
            f"- Modèle retenu : {CHAMPION_LABELS[result['champion']]}. Estimations des modèles (pb) : "
            + ", ".join(f"{MODEL_LABELS[k]} {fmt(v)}" for k, v in result["models"].items() if v is not None) + ".",
            "",
            "### Lecture descendante (contrôle)",
            "",
            f"- {fmt(result['top_down']['z_spread_bp'])} pb ; groupe : {result['top_down']['level_name']} ; "
            f"{result['top_down']['method']} ; écart de {fmt(result['top_down']['gap_pct'], 1)} % avec l'estimation.",
            "",
            md_table(pd.DataFrame(result["top_down"]["points"]), {
                "issuer": "Émetteur", "rating": "Notation", "seniority": "Séniorité", "currency": "Devise",
                "maturity_years": "Maturité (ans)", "z_spread_bp": "Spread observé (pb)",
                "corrected_z_spread_bp": "Spread corrigé (pb)"}),
            "",
            "### Comparables",
            "",
            _findings(comps),
            "",
            "### Contrôles",
            "",
            md_table(pd.DataFrame(result["controls"]), {"control": "Contrôle", "status": "Résultat", "detail": "Détail"}),
            "",
            "### Cascade (de l'obligation moyenne à l'estimation)",
            "",
            md_table(pd.DataFrame(result["waterfall"]), {"step": "Étape", "delta_bp": "Effet (pb)",
                                                         "cumulative_bp": "Cumul (pb)"}),
            "",
            "### Famille de l'obligation selon chaque méthode",
            "",
        ]
        for method, analysis in clusters.items():
            placement = analysis["tables"]["placement"] or {}
            out.append(f"- {METHODS[method]} : famille {placement.get('famille', '—')}"
                       + (f" ({placement['nom']})" if placement.get("nom") else "")
                       + (f", estimation au-dessus de {fmt(100 * placement['percentile'])} % de la famille"
                          if placement.get("percentile") is not None else "")
                       + (" ; obligation isolée de l'échantillon" if placement.get("isolee") else "") + ".")
        out += ["", "### Réserves de l'estimation", "", "\n".join(f"- {c}" for c in result["caveats"]), ""]

    if include_quality:
        from bondalorians.analysis.quality import model_quality

        quality = model_quality(estimator, include_forest=include_forest, max_cases=quality_cases)
        indicators = quality["tables"]["indicateurs"].assign(
            moins_de_10_pct=lambda d: 100 * d["moins_de_10_pct"], variation_entre_plis=lambda d: 100 * d["variation_entre_plis"])
        out += [
            "## 6. Qualité des modèles et validation",
            "",
            _findings(quality),
            "",
            "### Indicateurs hors échantillon",
            "",
            md_table(indicators, {
                "modele": "Modèle", "situation": "Situation", "cas": "Cas", "erreur_mediane_pct": "Erreur médiane (%)",
                "erreur_mediane_pb": "Erreur médiane (pb)", "biais_pct": "Biais (%)", "moins_de_10_pct": "Cas < 10 % (%)",
                "variation_entre_plis": "Variation entre plis (%)",
                "ratio_validation_apprentissage": "Validation / apprentissage"}),
            "",
            "### Tests",
            "",
            md_table(quality["tables"]["tests"], {"modele": "Modèle", "situation": "Situation", "test": "Test",
                                                  "statut": "Statut", "valeur": "Valeur", "regle": "Règle ou raison"}),
            "",
        ]

    out += [
        "## Limites connues du système",
        "",
        "- Les comparables et les modèles n'observent que la base : liquidité réelle, perspective de notation et "
        "actualité de l'émetteur sont invisibles.",
        "- Aucune prime de nouvelle émission n'est ajoutée.",
        "- Le code suppose une photographie à une date ; aucune dynamique temporelle n'est modélisée.",
        "- Les poids des comparables appris varient d'un jeu de données à l'autre ; ils ne s'interprètent pas un par un.",
        "",
    ]
    return "\n".join(out)
