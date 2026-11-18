"""Qualité de chaque modèle : indicateurs hors échantillon et tests de validation.

Protocole (le même pour tous les modèles) :
- validation croisée à 5 plis, sous deux schémas : « émetteur connu » (plis
  aléatoires) et « émetteur inconnu » (émetteurs entiers retirés) ;
- chaque obligation est estimée par des modèles entraînés sans elle (et sans
  son émetteur dans le second schéma) ;
- les indicateurs sont calculés sur ces estimations hors échantillon.

Tests appliqués à chaque modèle et chaque situation (seuils dans
`config/settings.yaml`, section `quality`) :
- biais : l'erreur médiane signée reste proche de zéro ;
- stabilité : l'erreur médiane varie peu d'un pli à l'autre ;
- apport : le modèle fait mieux que la courbe des pairs seule ;
- surapprentissage : l'erreur hors échantillon ne dépasse pas trop l'erreur
  d'apprentissage (modèles paramétriques seulement : pour une méthode à
  mémoire, l'obligation se retrouverait dans ses propres comparables) ;
- couverture : pour l'estimation retenue, l'intervalle à 90 % contient le
  vrai spread entre 85 % et 95 % des fois.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bondalorians.analysis.common import fmt, pct, real_bonds

UNSEEN = "emetteur_inconnu"
SITUATIONS = {"connu": "Émetteur connu", "inconnu": "Émetteur inconnu"}
DEFAULT_THRESHOLDS = {"max_bias": 0.02, "max_fold_cv": 0.15, "max_overfit_ratio": 1.3,
                      "coverage": [0.85, 0.95]}
MODELS = {  # colonne -> (libellé, paramétrique ?)
    "log_retenu": ("Estimation retenue", False),
    "peer_log": ("Courbe des pairs (P)", True),
    "log_a": ("Modèle A", False),
    "log_b": ("CatBoost (B)", True),
    "log_c": ("A corrigé (C)", False),
    "log_s": ("Ancrage senior (S)", False),
    "log_rf": ("Forêt aléatoire", True),
}
PASS, FAIL, NA = "réussi", "échoué", "non applicable"


def honest_coverage(data: pd.DataFrame, cal: dict, seed: int = 5) -> float:
    """Couverture de l'intervalle mesurée sur des obligations qui n'ont pas servi à le calibrer.

    L'intervalle est calibré sur les erreurs hors échantillon ; mesurer sa couverture sur
    ces mêmes erreurs donne presque 90 % par construction. Ici, les émetteurs sont coupés
    en deux moitiés : les quantiles sont recalculés sur l'une, la couverture est mesurée
    sur l'autre, puis l'inverse.
    """
    from bondalorians.quant.conformal import conformal_quantile

    issuers = data["issuer"].to_numpy()
    unique = np.unique(issuers)
    rng = np.random.default_rng(seed)
    half_of = dict(zip(unique, rng.permutation(len(unique)) % 2))
    halves = np.array([half_of[i] for i in issuers])
    scores = (data["err"] / data["sigma"]).to_numpy() if cal["adaptive"] else data["err"].to_numpy()
    alpha = cal["alpha"]
    covered = np.zeros(len(data), dtype=bool)
    keys = data["key"].to_numpy()
    for h in (0, 1):
        calib, test = halves != h, halves == h
        global_q = conformal_quantile(scores[calib], alpha)
        for key in np.unique(keys[test]):
            in_key = calib & (keys == key)
            q = conformal_quantile(scores[in_key], alpha) if in_key.sum() >= 50 else global_q
            mask = test & (keys == key)
            covered[mask] = scores[mask] <= q
    return float(covered.mean())


def _fold_ids(estimator) -> pd.DataFrame:
    folds = estimator.make_folds(np.random.default_rng(estimator.settings["calibration"]["seed"]))
    return pd.concat([pd.DataFrame({"row_id": np.arange(len(ids)), "scheme": scheme, "fold": ids})
                      for scheme, ids in folds.items()], ignore_index=True)


def _in_sample(estimator, columns: list[str], readers: list[str], n: int, seed: int) -> dict[str, float]:
    """Erreur médiane d'apprentissage (l'obligation fait partie des données d'entraînement)."""
    from bondalorians.quant.topdown import read

    df = estimator.df_
    rows = df.sample(min(n, len(df)), random_state=seed)
    y = np.log(rows["z_spread"].to_numpy())
    out = {}
    if "peer_log" in columns:
        out["peer_log"] = float(np.median(np.abs(estimator.peer_.predict_log(rows) - y)))
    if "log_b" in columns and estimator.ml_ is not None:
        out["log_b"] = float(np.median(np.abs(estimator.ml_.predict_log(rows) - y)))
    if "log_rf" in columns:
        from bondalorians.analysis.forest import fit_forest

        encoder, model = fit_forest(estimator)
        out["log_rf"] = float(np.median(np.abs(model.predict(encoder.transform(rows)) - y)))
    for reader in readers:
        values = [read(estimator, r, reader)["log"] for r in rows.to_dict("records")]
        out[f"log_{reader}"] = float(np.median(np.abs(np.array(values) - y)))
    return out


def model_quality(estimator, include_forest: bool = True, readers=("Descendante lissée", "Entonnoir avec maturité"),
                  max_cases: int | None = None, in_sample_cases: int = 600, seed: int = 5) -> dict:
    """Indicateurs et tests de chaque modèle. `max_cases` limite les lectures (plus lentes)."""
    if estimator.calibration_ is None:
        raise RuntimeError("Estimateur non calibré.")
    cal = estimator.calibration_
    thresholds = {**DEFAULT_THRESHOLDS, **estimator.settings.get("quality", {})}
    oof = cal["oof"].copy()
    oof = oof.merge(_fold_ids(estimator), on=["row_id", "scheme"], how="left")
    oof["issuer"] = estimator.df_["issuer"].to_numpy()[oof["row_id"].to_numpy()]
    champion_column = {"A": "log_a", "B": "log_b", "C": "log_c", "P": "peer_log", "S": "log_s"}
    oof["log_retenu"] = [
        float(row[champion_column[cal["champion"].get(row["key"], "A")]]) for _, row in oof.iterrows()
    ]
    models = dict(MODELS)
    if include_forest:
        from bondalorians.analysis.forest import forest_out_of_fold

        oof = oof.merge(forest_out_of_fold(estimator), on=["row_id", "scheme"], how="left")
    else:
        models.pop("log_rf")
    if readers:
        from bondalorians.quant.topdown import readings_out_of_fold

        readings = readings_out_of_fold(estimator, readers, max_cases=max_cases).drop(columns=["fold", "niveau"],
                                                                                       errors="ignore")
        oof = oof.merge(readings, on=["row_id", "scheme"], how="left")
        for reader in readers:
            models[f"log_{reader}"] = (f"Lecture : {reader.lower()}", False)
    models = {c: v for c, v in models.items() if c in oof and oof[c].notna().any()}
    oof["situation"] = np.where(oof["key"] == UNSEEN, "inconnu", "connu")
    truth_bp = np.exp(oof["y_log"])

    train = _in_sample(estimator, [c for c, (_, param) in models.items() if param], [], in_sample_cases, seed)
    indicators, tests = [], []
    for column, (label, parametric) in models.items():
        for situation, situation_label in SITUATIONS.items():
            data = oof[(oof["situation"] == situation) & oof[column].notna()]
            if data.empty:
                continue
            signed = data[column].astype(float) - data["y_log"]
            error = signed.abs()
            error_bp = (np.exp(data[column].astype(float)) - truth_bp[data.index]).abs()
            residual = ((data["y_log"] - data["y_log"].mean()) ** 2).sum()
            fold_medians = error.groupby(data["fold"]).median()
            cv = float(fold_medians.std(ddof=1) / fold_medians.mean()) if len(fold_medians) > 1 else float("nan")
            row = {
                "modele": label, "situation": situation_label, "cas": len(data),
                "erreur_mediane_pct": 100 * float(error.median()), "erreur_moyenne_pct": 100 * float(error.mean()),
                "erreur_mediane_pb": float(error_bp.median()), "biais_pct": 100 * float(signed.median()),
                "moins_de_5_pct": float((error < 0.05).mean()), "moins_de_10_pct": float((error < 0.10).mean()),
                "moins_de_20_pct": float((error < 0.20).mean()),
                "r2_hors_echantillon": float(1 - (signed**2).sum() / residual) if residual > 0 else float("nan"),
                "variation_entre_plis": cv,
                "erreur_apprentissage_pct": 100 * train[column] if column in train else float("nan"),
            }
            row["ratio_validation_apprentissage"] = (row["erreur_mediane_pct"] / row["erreur_apprentissage_pct"]
                                                     if column in train and train[column] > 0 else float("nan"))
            indicators.append(row)
            peer = oof[(oof["situation"] == situation)]
            peer_median = float((peer.loc[data.index, "peer_log"] - peer.loc[data.index, "y_log"]).abs().median())

            def add(test: str, status: str, value: str, rule: str) -> None:
                tests.append({"modele": label, "situation": situation_label, "test": test,
                              "statut": status, "valeur": value, "regle": rule})

            add("Biais", PASS if abs(row["biais_pct"]) <= 100 * thresholds["max_bias"] else FAIL,
                f"{fmt(row['biais_pct'], 1)} %", f"valeur absolue du biais ≤ {fmt(100 * thresholds['max_bias'])} %")
            add("Stabilité entre plis", PASS if cv <= thresholds["max_fold_cv"] else FAIL,
                f"{fmt(100 * cv)} %", f"variation ≤ {fmt(100 * thresholds['max_fold_cv'])} %")
            if column == "peer_log":
                add("Apport par rapport à la courbe des pairs", NA, "—", "c'est la référence")
            else:
                gain = 1 - row["erreur_mediane_pct"] / (100 * peer_median)
                add("Apport par rapport à la courbe des pairs", PASS if gain > 0 else FAIL,
                    f"{fmt(100 * gain, 1)} %", "erreur plus faible que la courbe seule")
            if parametric and situation == "connu" and column in train:
                ratio = row["ratio_validation_apprentissage"]
                add("Surapprentissage", PASS if ratio <= thresholds["max_overfit_ratio"] else FAIL,
                    f"× {fmt(ratio, 2)}", f"erreur validation / apprentissage ≤ {fmt(thresholds['max_overfit_ratio'], 1)}")
            else:
                add("Surapprentissage", NA, "—",
                    "émetteur inconnu : pas d'équivalent en apprentissage" if parametric
                    else "méthode à mémoire : en apprentissage, l'obligation serait son propre comparable")
            if column == "log_retenu":
                coverage = honest_coverage(data, cal, seed)
                low, high = thresholds["coverage"]
                row["couverture_intervalle"] = coverage
                add("Couverture de l'intervalle à 90 %", PASS if low <= coverage <= high else FAIL,
                    pct(coverage, 1), f"entre {fmt(100 * low)} et {fmt(100 * high)} %, mesurée sur des émetteurs "
                                      "non utilisés pour calibrer l'intervalle")

    indicators = pd.DataFrame(indicators)
    tests = pd.DataFrame(tests)
    by_tier = []
    for column, (label, _) in models.items():
        data = oof[oof[column].notna()]
        medians = (data[column].astype(float) - data["y_log"]).abs().groupby(data["key"]).median()
        by_tier.append({"modele": label, **{k: 100 * float(v) for k, v in medians.items()}})
    fold_detail = []
    for column, (label, _) in models.items():
        data = oof[oof[column].notna()]
        medians = (data[column].astype(float) - data["y_log"]).abs().groupby([data["situation"], data["fold"]]).median()
        fold_detail += [{"modele": label, "situation": SITUATIONS[s], "pli": int(f) + 1, "erreur_mediane_pct": 100 * v}
                        for (s, f), v in medians.items()]

    failed = tests[tests["statut"] == FAIL]
    retained = indicators[indicators["modele"] == "Estimation retenue"].set_index("situation")
    findings = [
        "Chaque obligation est estimée par des modèles entraînés sans elle (5 plis), et pour la situation « émetteur "
        "inconnu », sans aucune obligation de son émetteur. Tous les indicateurs portent sur ces estimations.",
        f"Estimation retenue : erreur médiane de {fmt(retained.loc['Émetteur connu', 'erreur_mediane_pct'], 1)} % "
        f"quand l'émetteur est connu, {fmt(retained.loc['Émetteur inconnu', 'erreur_mediane_pct'], 1)} % quand il "
        f"est inconnu ; {pct(retained.loc['Émetteur connu', 'moins_de_10_pct'])} des estimations à moins de 10 % "
        "d'erreur quand l'émetteur est connu.",
        f"{len(tests[tests['statut'] == PASS])} tests réussis, {len(failed)} échoués, "
        f"{len(tests[tests['statut'] == NA])} non applicables.",
    ]
    if not failed.empty:
        findings.append("Tests échoués : " + " ; ".join(
            f"{r.modele} ({r.situation.lower()}) : {r.test.lower()}, {r.valeur}" for r in failed.itertuples()) + ".")
    best = indicators.loc[indicators.groupby("situation")["erreur_mediane_pct"].idxmin()]
    findings += [f"{r.situation} : meilleur modèle {r.modele} ({fmt(r.erreur_mediane_pct, 1)} %)." for r in best.itertuples()]
    if readers and max_cases is not None:
        findings.append(f"Les lectures de contrôle sont évaluées sur un tirage de {fmt(max_cases)} cas, les autres "
                        "modèles sur tous les cas.")
    n_real = len(real_bonds(estimator))
    return {"title": "Qualité des modèles", "findings": findings, "thresholds": thresholds,
            "tables": {"indicateurs": indicators, "tests": tests, "par_palier": pd.DataFrame(by_tier),
                       "plis": pd.DataFrame(fold_detail), "obligations": n_real},
            "points": None}
