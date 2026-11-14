"""Onglet « Analyse » : comprendre l'échantillon, chaque modèle et le clustering.

Chaque bloc suit le même schéma : ce que montre le graphique, comment le lire,
le graphique, puis ce qu'il faut en retenir (constats calculés par
`bondalorians.analysis`, jamais rédigés à la main ici).
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from bondalorians.analysis import models as model_analysis
from bondalorians.analysis.clustering import ATYPICAL, METHOD_HELP, METHODS, cluster_universe
from bondalorians.analysis.common import BUCKET_ORDER, rating_bucket
from bondalorians.analysis.sample import describe_sample
from bondalorians.data.schema import RATING_SCALE, to_notch

# Palette de référence (validée pour les daltonismes), clair et sombre.
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"],
}
# Rampe séquentielle bleue, de la meilleure à la plus faible notation.
RAMP = {
    "light": ["#86b6ef", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"],
    "dark": ["#184f95", "#1c5cab", "#256abf", "#3987e5", "#5598e7", "#86b6ef", "#b7d3f6"],
}
INK = {"light": "#0b0b0b", "dark": "#f0efec"}
MUTED = "#898781"
TARGET = "Votre obligation"
MODEL_ORDER = ["Courbe des pairs", "A · Comparables", "B · CatBoost", "C · Comparables corrigés", "Forêt aléatoire"]


def _mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # noqa: BLE001 - détection impossible : thème clair
        return "light"


def _block(title: str, what: str, how: str, chart, findings: list[str], data=None) -> None:
    st.markdown(f"#### {title}")
    st.caption(f"**Ce que montre ce graphique.** {what}  \n**Comment le lire.** {how}")
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    if findings:
        st.info("**À retenir**\n\n" + "\n".join(f"- {f}" for f in findings))
    if data is not None:
        with st.expander("Voir les chiffres"):
            st.dataframe(data, hide_index=True, width="stretch")


def _axis(title: str, **kwargs) -> alt.Axis:
    return alt.Axis(title=title, gridColor="#e1e0d988", domain=False, ticks=False, **kwargs)


def _log_y(title: str = "Z-Spread (pb, échelle log)") -> alt.Y:
    return alt.Y("z_spread:Q", scale=alt.Scale(type="log", nice=False), axis=_axis(title))


def _target_layer(x_field: str, x_value, low, high, value, mode: str, x_type: str = "Q", sort=None):
    """Point de la cible et son intervalle (barre verticale)."""
    frame = pd.DataFrame([{x_field: x_value, "z_spread": value, "bas": low, "haut": high, "nom": TARGET}])
    x = alt.X(f"{x_field}:{x_type}", sort=sort) if sort else alt.X(f"{x_field}:{x_type}")
    rule = alt.Chart(frame).mark_rule(strokeWidth=2, color=INK[mode]).encode(
        x=x, y="bas:Q", y2="haut:Q")
    point = alt.Chart(frame).mark_point(shape="diamond", size=180, filled=True, color=INK[mode],
                                        stroke="white", strokeWidth=2).encode(
        x=x, y="z_spread:Q",
        tooltip=[alt.Tooltip("nom:N", title=""), alt.Tooltip("z_spread:Q", title="Estimation (pb)", format=".0f"),
                 alt.Tooltip("bas:Q", title="Bas de l'intervalle", format=".0f"),
                 alt.Tooltip("haut:Q", title="Haut de l'intervalle", format=".0f")])
    return rule + point


# ---------------------------------------------------------------- calculs
@st.cache_data(show_spinner="Analyse de l'échantillon…")
def _sample(_estimator, version: str) -> dict:
    return describe_sample(_estimator)


@st.cache_data(show_spinner="Analyse des modèles, forêt aléatoire comprise (environ 15 secondes)…")
def _models(_estimator, version: str) -> dict:
    return model_analysis.all_model_views(_estimator, include_forest=True)


@st.cache_data(show_spinner="Clustering de l'univers…")
def _clusters(_estimator, version: str, method: str, k, include_spread: bool, _target, target_key) -> dict:
    return cluster_universe(_estimator, method, k=k, include_spread=include_spread, target=_target)


# --------------------------------------------------------------- sections
def _sample_section(estimator, version: str, result: dict | None, mode: str) -> None:
    analysis = _sample(estimator, version)
    points, tables = analysis["points"], analysis["tables"]
    letters = [b for b in BUCKET_ORDER if b in set(points["lettre"])]
    ramp = RAMP[mode][: len(letters)]

    counts = tables["composition"]["lettre"].rename(columns={"valeur": "lettre"})
    chart = alt.Chart(counts).mark_bar(cornerRadiusEnd=4, color=SERIES[mode][0]).encode(
        x=alt.X("lettre:N", sort=letters, axis=_axis(None, labelAngle=0)),
        y=alt.Y("n:Q", axis=_axis("Nombre d'obligations")),
        tooltip=[alt.Tooltip("lettre:N", title="Notation"), alt.Tooltip("n:Q", title="Obligations")],
    ).properties(height=260)
    _block("Combien d'obligations par notation", "La répartition de l'échantillon par grande catégorie de notation.",
           "Une barre haute : beaucoup de comparables possibles dans cette catégorie, donc des estimations plus sûres.",
           chart, analysis["findings"][:1] + analysis["findings"][4:5])

    by_letter = tables["spread_par_lettre"]
    base = alt.Chart(by_letter).encode(x=alt.X("lettre:N", sort=letters, axis=_axis(None, labelAngle=0)))
    tip = [alt.Tooltip("lettre:N", title="Notation"), alt.Tooltip("n:Q", title="Obligations"),
           alt.Tooltip("p10:Q", title="10 % sous", format=".0f"), alt.Tooltip("mediane:Q", title="Médiane", format=".0f"),
           alt.Tooltip("p90:Q", title="10 % au-dessus de", format=".0f")]
    whisker = base.mark_rule(color=MUTED, strokeWidth=2).encode(
        y=alt.Y("p10:Q", scale=alt.Scale(type="log", nice=False), axis=_axis("Z-Spread (pb, échelle log)")), y2="p90:Q", tooltip=tip)
    box = base.mark_bar(size=26, cornerRadius=4, color=SERIES[mode][0], opacity=0.85).encode(
        y="p25:Q", y2="p75:Q", tooltip=tip)
    median = base.mark_tick(size=26, thickness=2, color=INK[mode]).encode(y="mediane:Q", tooltip=tip)
    chart = whisker + box + median
    if result:
        letter = rating_bucket(to_notch(result["bond"]["rating"]))
        low, high = result["interval_bp"] or [result["z_spread_bp"]] * 2
        chart = chart + _target_layer("lettre", letter, low, high, result["z_spread_bp"], mode, "N", letters)
    _block("Niveau des spreads par notation",
           "Pour chaque notation, où se situent les spreads de l'échantillon.",
           "Le rectangle contient la moitié centrale des obligations, le trait noir est la médiane, la ligne grise va "
           "du 10e au 90e centile. Échelle logarithmique : chaque graduation multiplie le spread."
           + (" Le losange est votre obligation, avec son intervalle." if result else ""),
           chart.properties(height=320), analysis["findings"][1:2], by_letter)

    scatter_points = points.sample(min(2500, len(points)), random_state=1)
    chart = alt.Chart(scatter_points).mark_circle(size=22, opacity=0.65).encode(
        x=alt.X("maturity_years:Q", axis=_axis("Maturité résiduelle (années)")),
        y=_log_y(),
        color=alt.Color("lettre:N", sort=letters, scale=alt.Scale(domain=letters, range=ramp),
                        legend=alt.Legend(title="Notation", orient="right")),
        tooltip=[alt.Tooltip("issuer:N", title="Émetteur"), alt.Tooltip("rating:N", title="Notation"),
                 alt.Tooltip("seniority:N", title="Séniorité"), alt.Tooltip("maturity_years:Q", title="Maturité", format=".1f"),
                 alt.Tooltip("z_spread:Q", title="Z-Spread (pb)", format=".0f")],
    )
    if result:
        low, high = result["interval_bp"] or [result["z_spread_bp"]] * 2
        chart = chart + _target_layer("maturity_years", result["bond"]["maturity_years"], low, high,
                                      result["z_spread_bp"], mode)
    _block("Spread, maturité et notation",
           "Chaque point est une obligation de l'échantillon.",
           "Plus on monte, plus le spread est élevé. Les couleurs foncées (notations faibles) sont plus haut ; "
           "à notation égale, le spread monte en général avec la maturité.",
           chart.properties(height=380), [])

    explained = tables["part_expliquee"].assign(part=lambda d: 100 * d["part_expliquee"])
    chart = alt.Chart(explained).mark_bar(cornerRadiusEnd=4, color=SERIES[mode][0]).encode(
        y=alt.Y("caracteristique:N", sort="-x", axis=_axis(None)),
        x=alt.X("part:Q", scale=alt.Scale(domain=[0, 100]), axis=_axis("Part de la dispersion expliquée (%)")),
        tooltip=[alt.Tooltip("caracteristique:N", title=""), alt.Tooltip("part:Q", title="Part expliquée (%)", format=".0f")],
    ).properties(height=260)
    _block("Qu'est-ce qui explique les spreads ?",
           "La part des écarts de spread entre obligations que chaque caractéristique explique à elle seule.",
           "100 % : connaître cette caractéristique suffirait à deviner le spread. 0 % : elle ne dit rien. "
           "Les parts ne s'additionnent pas, car les caractéristiques se recoupent.",
           chart, analysis["findings"][2:4], tables["part_expliquee"])

    cols = st.columns(3)
    for col, (key, label) in zip(cols, (("sector", "Secteur"), ("currency", "Devise"), ("seniority", "Séniorité"))):
        frame = tables["composition"][key]
        col.markdown(f"**{label}**")
        col.altair_chart(alt.Chart(frame).mark_bar(cornerRadiusEnd=3, color=SERIES[mode][0]).encode(
            y=alt.Y("valeur:N", sort="-x", axis=_axis(None, labelLimit=140, labelOverlap=False)),
            x=alt.X("n:Q", axis=_axis("Obligations")),
            tooltip=[alt.Tooltip("valeur:N", title=label), alt.Tooltip("n:Q", title="Obligations")],
        ).properties(height=max(120, 24 * len(frame))), width="stretch")


def _models_section(estimator, version: str, result: dict | None, mode: str) -> None:
    views = _models(estimator, version)

    view = views["courbe"]
    curves, bonds = view["points"]["curves"], view["points"]["bonds"]
    letters = list(model_analysis.LETTERS)
    ramp = RAMP[mode][1:6]
    color = alt.Color("lettre:N", sort=letters, scale=alt.Scale(domain=letters, range=ramp),
                      legend=alt.Legend(title="Notation"))
    dots = alt.Chart(bonds.sample(min(1500, len(bonds)), random_state=2)).mark_circle(size=14, opacity=0.35).encode(
        x=alt.X("maturity_years:Q", scale=alt.Scale(domain=[0, 30], clamp=True), axis=_axis("Maturité (années)")),
        y=_log_y(), color=color,
        tooltip=[alt.Tooltip("issuer:N", title="Émetteur"), alt.Tooltip("rating:N", title="Notation"),
                 alt.Tooltip("z_spread:Q", title="Z-Spread (pb)", format=".0f")])
    lines = alt.Chart(curves).mark_line(strokeWidth=2.5).encode(
        x="maturity_years:Q", y="z_spread:Q", color=color,
        tooltip=[alt.Tooltip("lettre:N", title="Notation"), alt.Tooltip("maturity_years:Q", title="Maturité", format=".1f"),
                 alt.Tooltip("z_spread:Q", title="Spread de la courbe (pb)", format=".0f")])
    chart = dots + lines
    if result:
        low, high = result["interval_bp"] or [result["z_spread_bp"]] * 2
        chart = chart + _target_layer("maturity_years", result["bond"]["maturity_years"], low, high, result["z_spread_bp"], mode)
    _block("1. Courbe des pairs : le spread « normal » d'un profil",
           "La première brique de l'estimation : une régression qui donne le spread attendu selon la notation et la maturité "
           "(et la séniorité, le secteur, la devise, fixés ici à leur valeur la plus fréquente).",
           "Les lignes sont les courbes apprises ; les points pâles, les obligations réelles de ces notations. "
           "Les points loin de leur courbe sont ceux que la suite du modèle doit expliquer.",
           chart.properties(height=380), view["findings"])

    view = views["emetteur"]
    issuers = view["points"].assign(brut=lambda d: 100 * d["ecart_brut"], retenu=lambda d: 100 * d["ecart_retenu"])
    if result:
        issuers["groupe"] = np.where(issuers["issuer"] == result["bond"]["issuer"], TARGET, "Autres émetteurs")
    else:
        issuers["groupe"] = "Autres émetteurs"
    extent = float(issuers[["brut", "retenu"]].abs().max().max())
    diagonal = alt.Chart(pd.DataFrame({"x": [-extent, extent]})).mark_line(color=MUTED, strokeDash=[4, 4]).encode(
        x="x:Q", y="x:Q")
    chart = diagonal + alt.Chart(issuers).mark_circle(opacity=0.7, stroke="white", strokeWidth=1).encode(
        x=alt.X("brut:Q", axis=_axis("Écart brut de l'émetteur à la courbe (%)")),
        y=alt.Y("retenu:Q", axis=_axis("Écart retenu dans l'estimation (%)")),
        size=alt.Size("n:Q", scale=alt.Scale(range=[20, 400]), legend=alt.Legend(title="Obligations")),
        color=alt.Color("groupe:N", scale=alt.Scale(domain=["Autres émetteurs", TARGET], range=[SERIES[mode][0], INK[mode]]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("issuer:N", title="Émetteur"), alt.Tooltip("n:Q", title="Obligations"),
                 alt.Tooltip("brut:Q", title="Écart brut (%)", format=".1f"),
                 alt.Tooltip("retenu:Q", title="Écart retenu (%)", format=".1f"),
                 alt.Tooltip("poids:Q", title="Poids", format=".0%")])
    _block("2. Écart propre à l'émetteur",
           "Chaque point est un émetteur : de combien ses obligations traitent au-dessus ou au-dessous de la courbe des pairs.",
           "Sur la diagonale, l'écart est retenu en entier ; plus un point s'en éloigne vers zéro, plus l'écart est réduit "
           "par prudence. Les petits points (peu d'obligations) sont les plus réduits.",
           chart.properties(height=360), view["findings"])

    if result:
        view = model_analysis.comparables_view(estimator, result)
        comps = view["points"]["comparables"]
        if not comps.empty:
            band = pd.DataFrame([{"bas": result["interval_bp"][0], "haut": result["interval_bp"][1]}]) if result["interval_bp"] else None
            chart = alt.Chart(comps).mark_circle(opacity=0.8, color=SERIES[mode][1], stroke="white", strokeWidth=1).encode(
                x=alt.X("distance:Q", scale=alt.Scale(zero=False), axis=_axis("Distance à votre obligation (0 = identique)")),
                y=alt.Y("adjusted_z_spread_bp:Q", scale=alt.Scale(type="log", zero=False),
                        axis=_axis("Spread ramené à votre profil (pb)")),
                size=alt.Size("weight:Q", scale=alt.Scale(range=[20, 500]), legend=alt.Legend(title="Poids")),
                tooltip=[alt.Tooltip("issuer:N", title="Émetteur"), alt.Tooltip("rating:N", title="Notation"),
                         alt.Tooltip("maturity_years:Q", title="Maturité"), alt.Tooltip("z_spread_bp:Q", title="Spread réel (pb)"),
                         alt.Tooltip("adjusted_z_spread_bp:Q", title="Ramené à votre profil (pb)"),
                         alt.Tooltip("weight:Q", title="Poids", format=".2f")])
            rule = alt.Chart(pd.DataFrame([{"y": result["z_spread_bp"]}])).mark_rule(color=INK[mode], strokeWidth=2).encode(y="y:Q")
            if band is not None:
                chart = alt.Chart(band).mark_rect(color=INK[mode], opacity=0.08).encode(y="bas:Q", y2="haut:Q") + chart
            _block("3. Les comparables de votre obligation",
                   "Les obligations d'autres émetteurs retenues comme comparables, avec le spread qu'elles auraient si elles "
                   "avaient exactement le profil de la vôtre.",
                   "À gauche, les plus ressemblantes ; les gros points pèsent le plus. Le trait horizontal est l'estimation, "
                   "la bande grise son intervalle.",
                   (chart + rule).properties(height=340), view["findings"], comps)
    else:
        st.markdown("#### 3. Les comparables de votre obligation")
        st.caption("Lancez une estimation dans l'onglet « Estimation » pour voir ses comparables ici.")

    view = views["reglage"]
    labels = ["Par défaut", "Appliqué"]
    chart = alt.Chart(view["points"]).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("critere:N", axis=_axis(None)), yOffset=alt.YOffset("reglage:N", sort=labels),
        x=alt.X("poids:Q", axis=_axis("Poids dans la distance", format="%")),
        color=alt.Color("reglage:N", sort=labels, scale=alt.Scale(domain=labels, range=[MUTED, SERIES[mode][0]]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("critere:N", title="Critère"), alt.Tooltip("reglage:N", title=""),
                 alt.Tooltip("poids:Q", title="Poids", format=".0%")]).properties(height=260)
    _block("4. Comment la ressemblance est mesurée",
           "Le poids de chaque critère quand on cherche des comparables : réglage par défaut et réglage appris sur les données.",
           "Un poids élevé : deux obligations qui diffèrent sur ce critère sont jugées très différentes.",
           chart, view["findings"])

    view = views["catboost"]
    if view["available"]:
        importance = view["points"]["importance"]
        bars = alt.Chart(importance).mark_bar(cornerRadiusEnd=4, color=SERIES[mode][0]).encode(
            y=alt.Y("variable:N", sort="-x", axis=_axis(None)),
            x=alt.X("effet_moyen_pct:Q", axis=_axis("Effet moyen sur le spread (%)")),
            tooltip=[alt.Tooltip("variable:N", title=""), alt.Tooltip("effet_moyen_pct:Q", title="Effet moyen (%)", format=".1f")],
        ).properties(height=280)
        dep = view["points"]["dependence"]
        order = [r for r in RATING_SCALE if r in set(dep["rating"])]
        strip = alt.Chart(dep).mark_circle(size=14, opacity=0.35, color=SERIES[mode][0]).encode(
            x=alt.X("rating:N", sort=order, axis=_axis("Notation", labelAngle=-45)),
            y=alt.Y("effet_pct:Q", axis=_axis("Effet de la notation sur le spread (%)")),
            tooltip=[alt.Tooltip("rating:N", title="Notation"), alt.Tooltip("effet_pct:Q", title="Effet (%)", format=".0f")])
        median = alt.Chart(dep).mark_line(color=INK[mode], strokeWidth=2).encode(
            x=alt.X("rating:N", sort=order), y="median(effet_pct):Q")
        c1, c2 = st.columns(2)
        with c1:
            _block("5. CatBoost (modèle B) : ce qu'il utilise",
                   "Un modèle d'apprentissage automatique qui prédit le spread sans connaître l'émetteur. "
                   "Mesure SHAP : la contribution de chaque variable à chaque prédiction.",
                   "Plus la barre est longue, plus la variable déplace les prédictions.", bars, view["findings"][:2])
        with c2:
            _block("Effet de la notation selon CatBoost",
                   "Pour chaque obligation, de combien sa notation pousse son spread prédit vers le haut ou le bas.",
                   "La ligne est la médiane. Elle monte de gauche à droite ; les petites baisses viennent des notations "
                   "rares (peu de points) et des interactions avec les autres variables.",
                   (strip + median).properties(height=280), view["findings"][2:])
    else:
        _block("5. CatBoost (modèle B)", "Indisponible.", "", None, view["findings"])

    view = views["correcteur"]
    if view["available"]:
        chart = alt.Chart(view["points"]).mark_bar(color=SERIES[mode][2], cornerRadiusEnd=2).encode(
            x=alt.X("correction_pct:Q", bin=alt.Bin(maxbins=40), axis=_axis("Correction appliquée au modèle A (%)")),
            y=alt.Y("count():Q", axis=_axis("Nombre d'obligations")),
            tooltip=[alt.Tooltip("count():Q", title="Obligations")]).properties(height=260)
        _block("6. Correcteur (modèle C)",
               "Un second CatBoost qui apprend les erreurs du modèle A et les corrige. Ici, l'ampleur de ses corrections.",
               "Une cloche étroite autour de 0 : il corrige peu. Il n'est utile que s'il améliore aussi l'erreur hors échantillon.",
               chart, view["findings"])

    view = views["foret"]
    chart = alt.Chart(view["points"]).mark_bar(cornerRadiusEnd=4, color=SERIES[mode][6]).encode(
        y=alt.Y("variable:N", sort="-x", axis=_axis(None)),
        x=alt.X("hausse_erreur_pct:Q", axis=_axis("Hausse de l'erreur si la variable est brouillée (points de %)")),
        tooltip=[alt.Tooltip("variable:N", title=""), alt.Tooltip("hausse_erreur_pct:Q", title="Hausse (points)", format=".1f")],
    ).properties(height=280)
    _block("7. Forêt aléatoire : un modèle de comparaison",
           "Une forêt aléatoire (des centaines d'arbres de décision moyennés), entraînée comme CatBoost. Elle ne sert pas "
           "à l'estimation : elle vérifie qu'un autre type de modèle ne ferait pas mieux.",
           "On brouille une variable et on mesure combien l'erreur augmente : plus elle augmente, plus la forêt en dépend.",
           chart, view["findings"])

    view = views["effets"]
    effects = view["points"]
    present = [m for m in MODEL_ORDER if m in set(effects["modele"])]
    palette = {m: SERIES[mode][i] for i, m in enumerate(MODEL_ORDER)}
    notch = effects[effects["variable"] == "notch"].assign(
        notation=lambda d: d["valeur"].map(lambda v: RATING_SCALE[int(v) - 1]))
    order = [RATING_SCALE[int(v) - 1] for v in sorted(notch["valeur"].unique())]
    st.markdown("#### 8. Ce que chaque modèle a appris, côte à côte")
    st.caption(f"**Ce que montre ce graphique.** {view['findings'][0]}  \n**Comment le lire.** Des courbes qui se "
               "superposent : les modèles sont d'accord. Des marches d'escalier : la forêt aléatoire découpe en tranches "
               "au lieu de lisser.")
    color = alt.Color("modele:N", sort=present, scale=alt.Scale(domain=present, range=[palette[m] for m in present]),
                      legend=alt.Legend(title=None, orient="top"))
    c1, c2 = st.columns(2)
    tip = [alt.Tooltip("modele:N", title="Modèle"), alt.Tooltip("z_spread:Q", title="Spread moyen (pb)", format=".0f")]
    c1.markdown("**Si toutes les obligations avaient cette notation**")
    c1.altair_chart(alt.Chart(notch).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=40)).encode(
        x=alt.X("notation:N", sort=order, axis=_axis("Notation", labelAngle=0, labelOverlap="greedy")),
        y=_log_y("Spread moyen prédit (pb, log)"),
        color=color, tooltip=[alt.Tooltip("notation:N", title="Notation")] + tip,
    ).properties(height=340), width="stretch")
    maturity = effects[effects["variable"] == "log_t"]
    c2.markdown("**Si toutes les obligations avaient cette maturité**")
    c2.altair_chart(alt.Chart(maturity).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=40)).encode(
        x=alt.X("axe:Q", scale=alt.Scale(type="log", domain=[1, 30]), axis=_axis("Maturité (années, log)")),
        y=_log_y("Spread moyen prédit (pb, log)"), color=color,
        tooltip=[alt.Tooltip("axe:Q", title="Maturité", format=".0f")] + tip,
    ).properties(height=340), width="stretch")
    st.info("**À retenir**\n\n" + "\n".join(f"- {f}" for f in view["findings"][1:]))


def _comparison_section(estimator, version: str, mode: str) -> None:
    views = _models(estimator, version)
    view = views["comparaison"]
    table = view["points"]
    present = [m for m in MODEL_ORDER if m in set(table["modele"])]
    palette = [SERIES[mode][MODEL_ORDER.index(m)] for m in present]
    tiers = ["Simple", "Intermédiaire", "Complexe", "Émetteur inconnu"]
    chart = alt.Chart(table).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("palier:N", sort=tiers, axis=_axis(None, labelAngle=0)),
        xOffset=alt.XOffset("modele:N", sort=present),
        y=alt.Y("erreur_mediane_bp:Q", axis=_axis("Erreur médiane (pb)")),
        color=alt.Color("modele:N", sort=present, scale=alt.Scale(domain=present, range=palette),
                        legend=alt.Legend(title="Modèle", orient="top")),
        tooltip=[alt.Tooltip("modele:N", title="Modèle"), alt.Tooltip("palier:N", title="Palier"),
                 alt.Tooltip("n:Q", title="Obligations testées"),
                 alt.Tooltip("erreur_mediane_bp:Q", title="Erreur médiane (pb)", format=".1f"),
                 alt.Tooltip("erreur_mediane_pct:Q", title="Erreur médiane (%)", format=".1f")],
    ).properties(height=340)
    _block("Quel modèle se trompe le moins ?",
           "L'erreur de chaque modèle sur des obligations qu'il n'a pas vues pendant l'apprentissage, par niveau de difficulté.",
           "Plus la barre est basse, mieux c'est. Comparez les barres à l'intérieur d'un même palier : les paliers n'ont pas "
           "les mêmes niveaux de spread.", chart, view["findings"], table)

    view = views["confiance"]
    deciles = view["points"]["deciles"].copy()
    deciles["tranche"] = deciles["decile"].map(lambda d: f"{d}")
    long = deciles.melt(id_vars=["tranche"], value_vars=["erreur_mediane_pct", "erreur_prevue_pct"],
                        var_name="mesure", value_name="erreur")
    long["mesure"] = long["mesure"].map({"erreur_mediane_pct": "Erreur réalisée", "erreur_prevue_pct": "Erreur prévue"})
    chart = alt.Chart(long).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=50)).encode(
        x=alt.X("tranche:N", sort=[str(i) for i in range(1, 11)],
                axis=_axis("Tranche de difficulté prévue (1 = 10 % les plus faciles, 10 = les plus difficiles)", labelAngle=0)),
        y=alt.Y("erreur:Q", axis=_axis("Erreur médiane (%)")),
        color=alt.Color("mesure:N", scale=alt.Scale(domain=["Erreur réalisée", "Erreur prévue"],
                                                    range=[SERIES[mode][0], SERIES[mode][1]]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("tranche:N", title="Tranche"), alt.Tooltip("mesure:N", title=""),
                 alt.Tooltip("erreur:Q", title="Erreur (%)", format=".1f")]).properties(height=320)
    _block("Le score de confiance tient-il ses promesses ?",
           "Les obligations du backtest classées de la plus facile à la plus difficile selon le modèle de difficulté, "
           "et l'erreur réellement commise dans chaque tranche.",
           "Si la courbe bleue monte de gauche à droite, le score sait distinguer les cas faciles des cas difficiles.",
           chart, view["findings"][:2], deciles)

    coverage = view["points"]["coverage"]
    bars = alt.Chart(coverage).mark_bar(cornerRadiusEnd=4, color=SERIES[mode][0], size=40).encode(
        x=alt.X("palier:N", sort=tiers, axis=_axis(None, labelAngle=0)),
        y=alt.Y("couverture:Q", scale=alt.Scale(domain=[0, 1]), axis=_axis("Part des vrais spreads dans l'intervalle", format="%")),
        tooltip=[alt.Tooltip("palier:N", title="Palier"), alt.Tooltip("couverture:Q", title="Couverture", format=".1%"),
                 alt.Tooltip("largeur_mediane_pct:Q", title="Largeur médiane (%)", format=".0f")])
    target = alt.Chart(pd.DataFrame([{"y": 0.9}])).mark_rule(color=INK[mode], strokeDash=[4, 4]).encode(y="y:Q")
    _block("L'intervalle à 90 % contient-il le vrai spread 9 fois sur 10 ?",
           "Pour chaque palier, la part des obligations du backtest dont le vrai spread tombe dans l'intervalle annoncé.",
           "La ligne pointillée marque 90 %. Une barre au niveau de la ligne : l'intervalle est honnête.",
           (bars + target).properties(height=280), view["findings"][2:], coverage)


def _clustering_section(estimator, version: str, result: dict | None, mode: str) -> None:
    st.markdown("#### Familles d'obligations (clustering)")
    st.caption(
        "Le clustering regroupe les obligations qui se ressemblent par leurs caractéristiques (notation, maturité, "
        "séniorité, taille, secteur, devise). **Il ne sert pas à estimer le spread** : il aide à lire l'échantillon et à "
        "voir dans quelle famille tombe votre obligation."
    )
    c1, c2, c3 = st.columns([2, 2, 1])
    method = c1.selectbox("Méthode", list(METHODS), format_func=METHODS.get)
    c1.caption(METHOD_HELP[method])
    automatic = c2.toggle("Nombre de familles automatique", value=True,
                          help="Choisi par la silhouette, qui mesure si les familles sont bien séparées.",
                          disabled=method == "hdbscan")
    k = None if automatic or method == "hdbscan" else c2.slider("Nombre de familles", 3, 8, 5)
    include_spread = c3.toggle("Avec le spread", value=False,
                               help="Forme les familles aussi sur le spread : des familles de risque, mais "
                                    "le lien familles-spread devient circulaire.")
    analysis = _clusters(estimator, version, method, k, include_spread, result,
                         None if result is None else result["run_id"])
    points, tables = analysis["points"].copy(), analysis["tables"]

    sizes = points[points["famille"] != ATYPICAL]["famille"].value_counts()
    shown = list(sizes.index[:7])
    profiles = tables["familles"].set_index("famille")
    names = {f: f"{f} · {profiles.loc[f, 'nom']}" for f in shown}
    points["groupe"] = points["famille"].map(lambda f: names.get(f, ATYPICAL if f == ATYPICAL else "Autres familles"))
    domain = [names[f] for f in shown] + [g for g in ("Autres familles", ATYPICAL) if g in set(points["groupe"])]
    palette = SERIES[mode][: len(shown)] + [MUTED if g == ATYPICAL else "#c3c2b7" for g in domain[len(shown):]]
    map_chart = alt.Chart(points.sample(min(3000, len(points)), random_state=3)).mark_circle(size=22, opacity=0.7).encode(
        x=alt.X("axe1:Q", axis=_axis(f"Axe 1 : surtout {tables['axes'][0]}")),
        y=alt.Y("axe2:Q", axis=_axis(f"Axe 2 : surtout {tables['axes'][1]}")),
        color=alt.Color("groupe:N", sort=domain, scale=alt.Scale(domain=domain, range=palette),
                        legend=alt.Legend(title="Famille", orient="right", labelLimit=320)),
        tooltip=[alt.Tooltip("groupe:N", title="Famille"), alt.Tooltip("issuer:N", title="Émetteur"),
                 alt.Tooltip("rating:N", title="Notation"), alt.Tooltip("maturity_years:Q", title="Maturité"),
                 alt.Tooltip("sector:N", title="Secteur"), alt.Tooltip("z_spread:Q", title="Z-Spread (pb)", format=".0f")])
    placement = tables["placement"]
    if placement:
        star = pd.DataFrame([{"axe1": placement["axe1"], "axe2": placement["axe2"], "nom": TARGET,
                              "famille": placement["famille"]}])
        map_chart = map_chart + alt.Chart(star).mark_point(shape="diamond", size=320, filled=True, color=INK[mode],
                                                           stroke="white", strokeWidth=2).encode(
            x="axe1:Q", y="axe2:Q", tooltip=[alt.Tooltip("nom:N", title=""), alt.Tooltip("famille:N", title="Famille")])
    _block(f"Carte des familles · {METHODS[method]}",
           "L'univers projeté sur deux axes qui résument au mieux les différences entre obligations ; une couleur par famille.",
           "Des couleurs en blocs nets : familles bien séparées. Des couleurs mêlées : familles qui se chevauchent. "
           + ("Le losange est votre obligation. " if placement else "")
           + (f"Seules les 7 plus grandes familles ont une couleur propre. " if len(sizes) > 7 else ""),
           map_chart.properties(height=440), analysis["findings"])

    profile_table = tables["familles"].assign(libelle=lambda d: d["famille"] + " · " + d["nom"])
    ordered = profile_table.sort_values("spread_median")["libelle"].tolist()
    base = alt.Chart(profile_table).encode(
        y=alt.Y("libelle:N", sort=ordered, axis=_axis(None, labelLimit=320)),
        tooltip=[alt.Tooltip("famille:N", title="Famille"), alt.Tooltip("nom:N", title=""),
                 alt.Tooltip("obligations:Q", title="Obligations"),
                 alt.Tooltip("spread_p25:Q", title="25 % sous", format=".0f"),
                 alt.Tooltip("spread_median:Q", title="Médiane", format=".0f"),
                 alt.Tooltip("spread_p75:Q", title="25 % au-dessus de", format=".0f")])
    box = base.mark_bar(cornerRadius=3, height=14, color=SERIES[mode][0], opacity=0.85).encode(
        x=alt.X("spread_p25:Q", scale=alt.Scale(type="log"), axis=_axis("Z-Spread (pb, log)")), x2="spread_p75:Q")
    tick = base.mark_tick(thickness=2, size=16, color=INK[mode]).encode(x="spread_median:Q")
    chart = box + tick
    if result:
        chart = chart + alt.Chart(pd.DataFrame([{"x": result["z_spread_bp"]}])).mark_rule(
            color=INK[mode], strokeDash=[4, 4], strokeWidth=2).encode(x="x:Q")
    _block("Niveau de spread de chaque famille",
           "Pour chaque famille, la moitié centrale de ses spreads (rectangle) et sa médiane (trait).",
           "Des rectangles bien étagés : les familles correspondent à des niveaux de risque différents. "
           + ("La ligne pointillée est l'estimation de votre obligation." if result else ""),
           chart.properties(height=max(220, 26 * len(profile_table))), [],
           profile_table.drop(columns="libelle"))

    c1, c2 = st.columns(2)
    with c1:
        share = tables["part_expliquee"]
        frame = pd.DataFrame([
            {"source": "Familles seules", "part": share["familles"]},
            {"source": "Lettre de notation seule", "part": share["lettre"]},
            {"source": "Familles et notation", "part": share["familles_et_lettre"]}])
        st.markdown("**Les familles expliquent-elles les spreads ?**")
        st.altair_chart(alt.Chart(frame).mark_bar(cornerRadiusEnd=4, color=SERIES[mode][0]).encode(
            y=alt.Y("source:N", sort=None, axis=_axis(None)),
            x=alt.X("part:Q", scale=alt.Scale(domain=[0, 1]), axis=_axis("Part de la dispersion des spreads expliquée", format="%")),
            tooltip=[alt.Tooltip("source:N", title=""), alt.Tooltip("part:Q", title="Part", format=".0%")],
        ).properties(height=200), width="stretch")
        st.caption("Si « Familles et notation » dépasse nettement « notation seule », les familles apportent une "
                   "information que la notation n'a pas.")
    with c2:
        if method != "hdbscan":
            sil = tables["silhouette"]
            chart = alt.Chart(sil).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60), color=SERIES[mode][0]).encode(
                x=alt.X("k:O", axis=_axis("Nombre de familles", labelAngle=0)),
                y=alt.Y("silhouette:Q", axis=_axis("Silhouette (séparation)")),
                tooltip=[alt.Tooltip("k:O", title="Familles"), alt.Tooltip("silhouette:Q", title="Silhouette", format=".2f")])
            st.markdown("**Combien de familles ? (k-means)**")
            st.altair_chart(chart.properties(height=200), width="stretch")
            st.caption("Silhouette : de −1 à 1. Au-dessus de 0,5, familles nettes ; sous 0,25, familles qui se chevauchent.")

    agreement = tables["accord"].melt(id_vars="methode", var_name="autre", value_name="accord")
    order = list(METHODS.values())
    heat = alt.Chart(agreement).encode(
        x=alt.X("autre:N", sort=order, axis=_axis(None, labelAngle=0)),
        y=alt.Y("methode:N", sort=order, axis=_axis(None)))
    rect = heat.mark_rect(cornerRadius=3).encode(
        color=alt.Color("accord:Q", scale=alt.Scale(domain=[0, 1], range=[RAMP[mode][0], RAMP[mode][-1]]),
                        legend=alt.Legend(title="Accord")),
        tooltip=[alt.Tooltip("methode:N", title=""), alt.Tooltip("autre:N", title="et"),
                 alt.Tooltip("accord:Q", title="Accord", format=".2f")])
    text = heat.mark_text(fontSize=13).encode(
        text=alt.Text("accord:Q", format=".2f"),
        color=alt.condition(alt.datum.accord > 0.5, alt.value("white"), alt.value(INK["light"])))
    _block("Les méthodes trouvent-elles les mêmes familles ?",
           "L'accord entre chaque paire de méthodes (indice de Rand ajusté).",
           "1 : les deux méthodes font exactement les mêmes familles ; 0 : pas plus d'accord que le hasard. "
           "Un faible accord rappelle que les familles dépendent de la méthode : ce ne sont pas des faits de marché.",
           (rect + text).properties(height=240), [])


def render(service, result: dict | None) -> None:
    estimator, version, mode = service.estimator, service.estimator.model_version_, _mode()
    if estimator.calibration_ is None:
        st.warning("Le moteur n'est pas calibré : relancez « Tout recalculer ».")
        return
    if result:
        st.caption(f"Votre dernière estimation est placée sur les graphiques : {result['bond']['issuer']}, "
                   f"{result['bond']['rating']}, {result['bond']['maturity_years']:.1f} ans, "
                   f"{result['z_spread_bp']:.0f} pb.")
    else:
        st.caption("Lancez une estimation dans l'onglet « Estimation » pour placer votre obligation sur les graphiques.")
    section = st.segmented_control(
        "Partie", ["Échantillon", "Modèles un par un", "Comparaison et confiance", "Clustering"],
        default="Échantillon", required=True, key="analysis_section", label_visibility="collapsed")
    if section == "Échantillon":
        _sample_section(estimator, version, result, mode)
    elif section == "Modèles un par un":
        _models_section(estimator, version, result, mode)
    elif section == "Comparaison et confiance":
        _comparison_section(estimator, version, mode)
    elif section == "Clustering":
        _clustering_section(estimator, version, result, mode)
