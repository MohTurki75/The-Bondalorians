"""Interface de démonstration.

Lancement : streamlit run app/main.py  (ou scripts/start.sh)
Toute la logique est dans `bondalorians.service.Service` ; cette page ne fait
qu'afficher ses résultats et ne calcule aucun chiffre.
"""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from bondalorians.config import load_settings
from bondalorians.data import repository
from bondalorians.data.schema import RATING_SCALE, SENIORITY_RANK
from bondalorians.data.stores import JsonlStore, PostgresStore
from bondalorians.service import Service

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "data" / "synthetic_bonds.csv"
MODEL_LABELS = {
    "courbe_pairs_bp": "Courbe des pairs seule",
    "A_comparables_bp": "A · Comparables ajustés",
    "B_catboost_bp": "B · CatBoost indépendant",
    "C_comparables_corriges_bp": "C · Comparables corrigés",
}
CHAMPION_KEY = {"A": "A_comparables_bp", "B": "B_catboost_bp", "C": "C_comparables_corriges_bp"}
TIER_LABELS = {
    "simple": "Simple", "intermediaire": "Intermédiaire", "complexe": "Complexe",
    "emetteur_inconnu": "Émetteur inconnu",
}
BACKTEST_MODELS = {
    "courbe_pairs": "courbe des pairs", "A_comparables": "modèle A",
    "B_catboost": "modèle B", "C_comparables_corriges": "modèle C",
}
STATUS_LABEL = {"ok": "✓ Conforme", "alerte": "⚠ Alerte", "echec": "✕ Échec"}
# Polarité de la cascade : hausse et baisse du spread, totaux en neutre.
COLORS = {"Hausse": "#e34948", "Baisse": "#2a78d6", "Niveau": "#898781"}

st.set_page_config(page_title="The Bondalorians · Z-Spread juste", layout="wide")


# ----------------------------------------------------------------- service
@st.cache_resource(show_spinner="Préparation des modèles (jusqu'à une minute avec CatBoost)…")
def load_service() -> tuple[Service, str]:
    config_dir = Path(os.environ.get("BONDALORIANS_CONFIG_DIR", ROOT / "config"))
    settings = load_settings(config_dir / "settings.yaml")
    use_ml = os.environ.get("BONDALORIANS_USE_ML", "1") != "0"
    dsn = os.environ.get("BONDALORIANS_DSN")
    if dsn:
        schema_map = repository.load_schema_map(config_dir / "schema_map.yaml")
        service = Service(
            repository.load_universe_postgres(dsn, schema_map), settings, use_ml=use_ml,
            run_store=PostgresStore(dsn, "app.runs", "run_id"),
            feedback_store=PostgresStore(dsn, "app.feedback", "feedback_id"),
        )
        return service, "PostgreSQL"
    csv_path = Path(os.environ.get("BONDALORIANS_CSV", SYNTHETIC))
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    service = Service(
        repository.load_universe_csv(csv_path), settings, use_ml=use_ml,
        run_store=JsonlStore(ROOT / "runs" / "runs.jsonl"),
        feedback_store=JsonlStore(ROOT / "runs" / "feedback.jsonl"),
    )
    label = "Données synthétiques" if csv_path.resolve() == SYNTHETIC.resolve() else f"Fichier {csv_path.name}"
    return service, label


try:
    service, source = load_service()
except FileNotFoundError as missing:
    st.error(
        f"Aucune source de données : `{missing}` est introuvable. Définissez `BONDALORIANS_DSN` "
        "(PostgreSQL) ou `BONDALORIANS_CSV`, ou générez des données d'essai avec "
        "`python scripts/make_synthetic.py`. Voir docs/LANCEMENT.md."
    )
    st.stop()

universe = service.estimator.df_
real = universe[~universe["isin"].astype(str).str.startswith("FB_")]


# ---------------------------------------------------------------- affichage
def waterfall_chart(steps: list[dict], final_bp: float) -> alt.Chart:
    rows, previous = [], 0.0
    for i, step in enumerate(steps):
        if step["delta_bp"] is None:
            rows.append({"etape": step["step"], "debut": 0.0, "fin": step["cumulative_bp"],
                         "sens": "Niveau", "texte": f"{step['cumulative_bp']:.0f} pb"})
        else:
            delta = step["delta_bp"]
            rows.append({"etape": step["step"], "debut": previous, "fin": step["cumulative_bp"],
                         "sens": "Hausse" if delta >= 0 else "Baisse",
                         "texte": f"{delta:+.0f} pb"})
        previous = step["cumulative_bp"]
    rows.append({"etape": "Z-Spread juste estimé", "debut": 0.0, "fin": final_bp,
                 "sens": "Niveau", "texte": f"{final_bp:.0f} pb"})
    data = pd.DataFrame(rows)
    data["etiquette_x"] = data[["debut", "fin"]].max(axis=1)
    order = data["etape"].tolist()
    y = alt.Y("etape:N", sort=order, title=None, axis=alt.Axis(labelLimit=330, ticks=False, domain=False))
    bars = alt.Chart(data).mark_bar(cornerRadius=3, height=16).encode(
        y=y,
        x=alt.X("debut:Q", title="Z-Spread (pb)", axis=alt.Axis(gridColor="#e1e0d9", domain=False, ticks=False)),
        x2="fin:Q",
        color=alt.Color(
            "sens:N", title="Effet sur le spread",
            scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
            legend=alt.Legend(orient="top"),
        ),
        tooltip=[alt.Tooltip("etape:N", title="Étape"), alt.Tooltip("texte:N", title="Effet"),
                 alt.Tooltip("fin:Q", title="Cumul (pb)", format=".1f")],
    )
    labels = alt.Chart(data).mark_text(align="left", dx=6, color="#52514e").encode(
        y=y, x="etiquette_x:Q", text="texte:N",
    )
    return (bars + labels).properties(height=30 * len(data) + 40)


def show_result(result: dict) -> None:
    conf, interval = result["confidence"], result["interval_bp"]
    cols = st.columns(5 if result["observed_z_spread_bp"] else 4)
    cols[0].metric("Z-Spread juste", f"{result['z_spread_bp']:.0f} pb")
    cols[1].metric(
        f"Intervalle à {100 * result['interval_level']:.0f} %",
        f"{interval[0]:.0f} – {interval[1]:.0f} pb" if interval else "indisponible",
    )
    cols[2].metric(
        "Confiance", f"{conf['score']:.0f}/100 · {conf['grade']}" if conf["score"] is not None else "—",
        help="Part des cas du backtest jugés plus difficiles que celui-ci.",
    )
    cols[3].metric("Palier", TIER_LABELS[result["tier"]], help=result["tier_reason"])
    if result["observed_z_spread_bp"]:
        cols[4].metric(
            "Spread observé", f"{result['observed_z_spread_bp']:.0f} pb",
            help="Spread de cette obligation dans l'univers. Il n'a pas servi à l'estimation.",
        )
    st.caption(f"{result['tier_reason']}  ·  Estimation `{result['run_id']}`")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("D'où vient ce chiffre")
        st.altair_chart(waterfall_chart(result["waterfall"], result["z_spread_bp"]), width="stretch")
        with st.expander("Voir la cascade en tableau"):
            st.dataframe(
                pd.DataFrame(result["waterfall"]).rename(
                    columns={"step": "Étape", "delta_bp": "Effet (pb)", "cumulative_bp": "Cumul (pb)"}
                ), hide_index=True, width="stretch",
            )
    with right:
        st.subheader("Modèles")
        models = pd.DataFrame(
            [{"Modèle": MODEL_LABELS[k], "Z-Spread (pb)": v,
              "Retenu": "✓" if k == CHAMPION_KEY[result["champion"]] else ""}
             for k, v in result["models"].items() if v is not None]
        )
        st.dataframe(models, hide_index=True, width="stretch")
        st.subheader("Contrôles")
        controls = pd.DataFrame(result["controls"])
        controls["status"] = controls["status"].map(STATUS_LABEL)
        st.dataframe(
            controls.rename(columns={"control": "Contrôle", "status": "Résultat", "detail": "Détail"}),
            hide_index=True, width="stretch",
        )

    offset = result["issuer_offset"]
    st.subheader(f"Comparables hors émetteur ({len(result['comparables'])})")
    st.caption(
        f"Filtre appliqué : {result['filter_level']}. Courbe émetteur : {offset['n_bonds']} obligation(s), "
        f"retenue à {100 * offset['lambda']:.0f} %."
    )
    if result["comparables"]:
        comparables = pd.DataFrame(result["comparables"]).rename(columns={
            "isin": "ISIN", "issuer": "Émetteur", "rating": "Notation", "seniority": "Séniorité",
            "sector": "Secteur", "maturity_years": "Maturité (ans)", "z_spread_bp": "Z-Spread (pb)",
            "adjusted_z_spread_bp": "Ajusté à la cible (pb)", "distance": "Distance", "weight": "Poids",
        })
        st.dataframe(comparables, hide_index=True, width="stretch", height=280)

    with st.expander("Explication rédigée et réserves", expanded=True):
        st.text(result["narrative"])


def feedback_form(result: dict) -> None:
    st.subheader("Votre retour")
    with st.form("feedback", clear_on_submit=True):
        kind = st.radio(
            "Nature du retour", ["trade", "opinion"], horizontal=True,
            format_func={"trade": "Spread réellement traité (poids 1)",
                         "opinion": "Avis d'expert (poids 0,3)"}.get,
        )
        col1, col2 = st.columns([1, 3])
        spread = col1.number_input("Z-Spread (pb)", min_value=1.0, value=float(result["z_spread_bp"]), step=1.0)
        reason = col2.text_input("Motif", placeholder="Ex. : traité avec le broker X, liquidité faible…")
        if st.form_submit_button("Enregistrer le retour"):
            saved = service.submit_feedback(result["run_id"], kind, spread, reason)
            st.session_state["notice"] = (
                f"Retour {saved['feedback_id']} enregistré. Il pèse dès la prochaine estimation ; "
                "relancez l'estimation pour voir son effet."
            )
            st.rerun()


# --------------------------------------------------------------------- page
st.title("Z-Spread juste d'une obligation")
st.caption(
    f"Source : {source} · {len(real)} obligations, {real['issuer'].nunique()} émetteurs · "
    f"version {service.estimator.model_version_}"
)
if source == "Données synthétiques":
    st.warning("Données synthétiques : les chiffres valident la chaîne, pas la précision réelle.")

with st.sidebar:
    st.header("Modèles")
    report = service.calibration_report()
    st.metric("k du poids émetteur", f"{report['issuer_shrinkage']['k']:.2f}",
              help="lambda = n / (n + k). Estimé par analyse de variance des résidus.")
    tuning = report["comparables_tuning"]
    if tuning:
        st.write(
            f"Comparables réglés : {'adoptés' if tuning['adopted'] else 'non adoptés'} "
            f"(erreur de validation {tuning['valid_error_pct']['default']} % → "
            f"{tuning['valid_error_pct']['tuned']} %)."
        )
    st.write("CatBoost : " + ("actif" if service.estimator.ml_ is not None else "inactif"))
    st.write(f"Retours intégrés : {len(universe) - len(real)}")
    if st.button("Tout recalculer", help="Relit les données et relance le backtest complet."):
        load_service.clear()
        st.session_state.clear()
        st.rerun()

tab_estimate, tab_backtest, tab_log = st.tabs(["Estimation", "Backtest et calibrage", "Journal d'audit"])

with tab_estimate:
    if "notice" in st.session_state:
        st.success(st.session_state.pop("notice"))
    mode = st.radio("Obligation à estimer", ["Par ses caractéristiques", "Par son ISIN (déjà dans l'univers)"],
                    horizontal=True)
    bond = None
    if mode.startswith("Par son ISIN"):
        labels = dict(zip(real["isin"], real["issuer"] + " · " + real["rating"].astype(str)))
        isin = st.selectbox("ISIN", list(labels), format_func=lambda i: f"{i} — {labels[i]}")
        if st.button("Estimer", type="primary"):
            bond = isin
    else:
        # Hors du formulaire : le choix de l'émetteur préremplit son secteur et son pays.
        issuer = st.selectbox(
            "Émetteur", sorted(real["issuer"].unique()), accept_new_options=True,
            help="Choisissez un émetteur de l'univers ou saisissez un nouveau nom.",
        )
        known = real[real["issuer"] == issuer]
        sectors, countries = sorted(real["sector"].unique()), sorted(real["country"].unique())
        if known.empty:
            st.caption("Émetteur absent de l'univers : le cas sera traité comme complexe.")
        else:
            st.caption(f"Émetteur connu : {len(known)} obligation(s) dans l'univers.")
        with st.form("bond"):
            c2, c3 = st.columns(2)
            sector = c2.selectbox(
                "Secteur", sectors, index=0 if known.empty else sectors.index(known["sector"].iat[0]),
            )
            country = c3.selectbox(
                "Pays", countries, index=0 if known.empty else countries.index(known["country"].iat[0]),
            )
            c4, c5, c6 = st.columns(3)
            currency = c4.selectbox("Devise", sorted(real["currency"].unique()))
            rating = c5.selectbox("Notation", RATING_SCALE[:19], index=8)
            seniority = c6.selectbox(
                "Séniorité", list(SENIORITY_RANK),
                index=list(SENIORITY_RANK).index("SENIOR_UNSECURED"),
            )
            c7, c8, c9 = st.columns(3)
            maturity = c7.number_input("Maturité résiduelle (ans)", min_value=0.1, max_value=50.0, value=5.0, step=0.5)
            coupon = c8.selectbox("Type de coupon", sorted(real["coupon_type"].unique()))
            callable_ = c9.checkbox("Remboursable par anticipation")
            if st.form_submit_button("Estimer", type="primary"):
                if not issuer:
                    st.error("Indiquez un émetteur.")
                else:
                    bond = {
                        "issuer": issuer, "sector": sector, "country": country, "currency": currency,
                        "rating": rating, "seniority": seniority, "maturity_years": maturity,
                        "coupon_type": coupon, "callable": callable_,
                    }
    if bond is not None:
        try:
            st.session_state["result"] = service.estimate(bond)
        except (KeyError, ValueError) as error:
            st.session_state.pop("result", None)
            st.error(f"Estimation impossible : {error}")
    if "result" in st.session_state:
        st.divider()
        show_result(st.session_state["result"])
        feedback_form(st.session_state["result"])

with tab_backtest:
    st.subheader("Erreur hors échantillon par palier")
    st.caption(
        "Chaque obligation est estimée sans avoir servi à l'apprentissage. "
        "« emetteur_inconnu » : tout l'émetteur est retiré."
    )
    backtest = pd.DataFrame(service.backtest_summary())
    backtest["palier"] = backtest["palier"].map(TIER_LABELS)
    names = {"palier": "Palier", "n": "Obligations testées", "champion": "Modèle retenu"}
    for key, label in BACKTEST_MODELS.items():
        names[f"mediane_abs_bp_{key}"] = f"Erreur médiane, {label} (pb)"
        names[f"mae_bp_{key}"] = f"Erreur moyenne, {label} (pb)"
    ordered = [c for c in names if c in backtest.columns]
    st.dataframe(backtest[ordered].rename(columns=names), hide_index=True, width="stretch")
    st.subheader("Paramètres appris")
    st.json(report, expanded=2)

with tab_log:
    runs = service.run_store.all()
    st.subheader(f"Estimations enregistrées ({len(runs)})")
    if runs:
        table = pd.DataFrame([
            {"Identifiant": r["id"], "Date": r["created_at"],
             "Émetteur": r["payload"]["result"]["bond"]["issuer"],
             "Z-Spread (pb)": r["payload"]["result"]["z_spread_bp"],
             "Palier": TIER_LABELS[r["payload"]["result"]["tier"]],
             "Version": r["payload"]["result"]["model_version"]}
            for r in reversed(runs[-200:])
        ])
        st.dataframe(table, hide_index=True, width="stretch", height=260)
        chosen = st.selectbox("Rejouer une estimation", table["Identifiant"])
        st.json(service.get_run(chosen)["payload"], expanded=1)
    else:
        st.info("Aucune estimation pour l'instant.")
    feedback = service.feedback_store.all()
    st.subheader(f"Retours utilisateurs ({len(feedback)})")
    if feedback:
        st.dataframe(pd.DataFrame([
            {"Identifiant": f["id"], "Date": f["created_at"], "Nature": f["payload"]["kind"],
             "Z-Spread saisi (pb)": f["payload"]["z_spread_bp"],
             "Estimation d'origine (pb)": f["payload"]["estimate_bp"],
             "Émetteur": f["payload"]["bond"]["issuer"], "Motif": f["payload"]["reason"]}
            for f in reversed(feedback)
        ]), hide_index=True, width="stretch")
