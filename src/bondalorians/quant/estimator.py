"""Estimateur du Z-Spread juste.

Modèle A (transparent), en log :
    spread = courbe des pairs + lambda * écart émetteur + (1 - lambda) * écart des comparables
    lambda = n / (n + k), k estimé par analyse de variance.
Modèle B : CatBoost indépendant (challenger, contrôle de désaccord).
Modèle C : modèle A corrigé par un CatBoost qui apprend son erreur résiduelle.

Le backtest désigne le champion par palier, entraîne le modèle de difficulté
et calibre l'intervalle conforme.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

from bondalorians import __version__
from bondalorians.config import load_settings
from bondalorians.data.schema import DESCRIPTIVE_COLUMNS, prepare, prepare_target
from bondalorians.domain.controls import run_controls
from bondalorians.domain.tiers import classify
from bondalorians.explain.narrative import STANDING_CAVEATS, build_narrative
from bondalorians.explain.waterfall import build_waterfall
from bondalorians.quant.confidence import (
    ERROR_FEATURES, ErrorModel, confidence_from_difficulty, error_features,
)
from bondalorians.quant.conformal import conformal_quantile
from bondalorians.quant.gbm import SpreadCorrector, SpreadGBM, catboost_available
from bondalorians.quant.peer_curve import PeerCurve
from bondalorians.quant.subordination import SubordinationModel
from bondalorians.quant.topdown import read as read_control
from bondalorians.quant.shrinkage import estimate_shrinkage_k, estimate_slope_shrinkage
from bondalorians.quant.tuning import WEIGHT_KEYS, tune_comparables

UNSEEN_KEY = "emetteur_inconnu"
GLOBAL_KEY = "_global"
FILTER_LEVELS = [
    "devise, séniorité et type de coupon",
    "devise et séniorité",
    "devise seule",
    "aucun filtre dur",
]
# P : courbe des pairs seule. Elle concourt aussi : pour un émetteur inconnu, les
# comparables d'autres émetteurs peuvent ajouter plus de bruit que d'information.
# S : dette subordonnée ancrée sur la dette senior de l'émetteur (égale à A ailleurs).
MODEL_COLUMNS = {"A": "log_a", "B": "log_b", "C": "log_c", "P": "peer_log", "S": "log_s"}
_ARRAYS_OBJECT = ["isin", "issuer", "currency", "seniority", "coupon_type", "sector", "country"]
_ARRAYS_FLOAT = ["notch", "log_t", "maturity_years", "log_amt", "sen_rank", "obs_weight", "z_spread"]


class ZSpreadEstimator:
    def __init__(self, settings: dict | None = None, use_ml: bool = True,
                 comparables_params: dict | None = None):
        self.settings = settings or load_settings()
        self.use_ml = use_ml and catboost_available()
        cfg = self.settings["comparables"]
        self.params_ = comparables_params or {
            "weights": dict(cfg["weights"]), "bandwidth": cfg["bandwidth"], "top_k": cfg["top_k"],
        }
        self.calibration_: dict | None = None
        self.tuning_: dict | None = None
        self.ml_ = self.corrector_ = None

    def spawn(self, use_ml: bool | None = None) -> "ZSpreadEstimator":
        """Estimateur vierge héritant des paramètres de comparables (pour les plis)."""
        return ZSpreadEstimator(
            self.settings, use_ml=self.use_ml if use_ml is None else use_ml,
            comparables_params=self.params_,
        )

    # ------------------------------------------------------------------ fit
    def fit(self, universe: pd.DataFrame, calibrate: bool = True) -> "ZSpreadEstimator":
        """Ajuste tout. Avec `calibrate`, règle les comparables puis lance le backtest."""
        df = prepare(universe, require_target=True)
        self.dropped_ = df.attrs.get("dropped", 0)
        self.fit_base(df)
        cfg = self.settings["comparables"]
        if calibrate and cfg["tune"]:
            self.tuning_ = tune_comparables(
                self, cfg["tune_trials"], cfg["tune_share"], self.settings["calibration"]["seed"],
                cfg["tune_min_weight"],
            )
        self.fit_ml()
        if calibrate:
            self.calibration_ = self._calibrate()
        return self

    def refit(self, universe: pd.DataFrame) -> "ZSpreadEstimator":
        """Réajustement rapide après un retour utilisateur.

        La courbe des pairs, les résidus et k sont recalculés. Les modèles
        CatBoost, les paramètres de comparables et la calibration sont conservés
        jusqu'au prochain `fit` complet.
        """
        self.fit_base(prepare(universe, require_target=True))
        return self

    def fit_base(self, df: pd.DataFrame) -> None:
        self.df_ = df.reset_index(drop=True)
        curve = self.settings["peer_curve"]
        self.peer_ = PeerCurve(curve["ridge"], enriched=curve.get("enriched", False)).fit(self.df_)
        self.a_ = {c: self.df_[c].to_numpy(dtype=object) for c in _ARRAYS_OBJECT}
        self.a_.update({c: self.df_[c].to_numpy(dtype=float) for c in _ARRAYS_FLOAT})
        self.a_["resid"] = np.log(self.a_["z_spread"]) - self.peer_.predict_log(self.df_)
        k_setting = self.settings["comparables"]["issuer_shrinkage_k"]
        if k_setting == "auto":
            self.shrinkage_ = estimate_shrinkage_k(self.a_["resid"], self.a_["issuer"])
        else:
            self.shrinkage_ = {"k": float(k_setting), "estimated": False}
        self.k_ = self.shrinkage_["k"]
        noise_sd = self.shrinkage_.get("noise_sd")
        self.noise_sd_ = float(noise_sd) if noise_sd else float(np.std(self.a_["resid"]))
        self.slope_ = {"k_b": None, "estimated": False}
        if self.settings["comparables"].get("issuer_slope", False):
            self.slope_ = estimate_slope_shrinkage(
                self.a_["resid"], self.a_["issuer"], self.a_["log_t"], self.noise_sd_**2)
        self.subordination_ = None
        if self.settings.get("subordination", {}).get("enabled", True):
            self.subordination_ = SubordinationModel().fit(self)
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(self.df_[["issuer", "maturity_years", "z_spread"]]).to_numpy().tobytes()
        ).hexdigest()[:10]
        self.model_version_ = f"{__version__}-{len(self.df_)}-{digest}"

    def fit_ml(self) -> None:
        self.ml_ = self.corrector_ = None
        if not self.use_ml:
            return
        ml = self.settings["ml"]
        self.ml_ = SpreadGBM(ml["iterations"], ml["depth"], ml["learning_rate"],
                             monotone=ml.get("monotone", False)).fit(self.df_)
        if not ml["corrector"]:
            return
        # Le correcteur apprend l'erreur du modèle A calculée sans l'obligation
        # elle-même. Chaque obligation est vue dans les deux situations réelles :
        # émetteur connu (elle seule retirée) et émetteur inconnu (émetteur retiré).
        rows = self.df_.to_dict("records")
        peer_logs = self.peer_.predict_log(self.df_)
        y = np.log(self.a_["z_spread"])
        cores = [
            self._core(row, exclude_isin=row["isin"], exclude_issuer=unseen, peer_log=float(pl))
            for unseen in (False, True) for row, pl in zip(rows, peer_logs)
        ]
        frame = pd.concat([self.df_, self.df_], ignore_index=True)
        residual = np.concatenate([y, y]) - np.array([c["log_a"] for c in cores])
        weights = np.concatenate([self.a_["obs_weight"], self.a_["obs_weight"]]) / 2
        self.corrector_ = SpreadCorrector(
            ml["corrector_iterations"], ml["corrector_depth"]
        ).fit(frame, cores, residual, weights)

    # ----------------------------------------------------------------- cœur
    def candidates(self, t: dict, exclude_isin=None, exclude_issuer: bool = False,
                   peer_log: float | None = None) -> dict:
        """Tout ce qui ne dépend pas des paramètres de comparables."""
        a, cfg = self.a_, self.settings["comparables"]
        keep = np.ones(len(self.df_), dtype=bool)
        if exclude_isin is not None:
            keep &= a["isin"] != exclude_isin
        if peer_log is None:
            peer_log = float(self.peer_.predict_log(pd.DataFrame([t]))[0])
        same_issuer = a["issuer"] == t["issuer"]

        # Écart émetteur, rétréci selon le nombre d'obligations.
        issuer_mask = keep & same_issuer & (not exclude_issuer)
        offset_mask = issuer_mask & (a["currency"] == t["currency"])
        if not offset_mask.any():
            offset_mask = issuer_mask
        n_issuer = float(a["obs_weight"][offset_mask].sum())
        r_issuer, slope, lam_b, delta_t, issuer_disp = 0.0, 0.0, 0.0, 0.0, float("nan")
        if n_issuer > 0:
            w, r, lt = a["obs_weight"][offset_mask], a["resid"][offset_mask], a["log_t"][offset_mask]
            r_issuer = float(np.average(r, weights=w))
            k_b = self.slope_.get("k_b")
            if k_b is not None and np.isfinite(k_b) and len(r) >= 2:
                center = float(np.average(lt, weights=w))
                sxx = float((w * (lt - center) ** 2).sum())
                if sxx > 1e-9:
                    slope = float((w * (lt - center) * (r - r_issuer)).sum() / sxx)
                    lam_b = sxx / (sxx + k_b)
                    # Pas d'extrapolation lointaine : au plus 0,3 en log au-delà des maturités de l'émetteur.
                    delta_t = float(np.clip(t["log_t"] - center, lt.min() - center - 0.3, lt.max() - center + 0.3))
            if len(r) >= 2:
                fitted = r_issuer + lam_b * slope * (lt - (np.average(lt, weights=w) if lam_b else 0.0))
                issuer_disp = float(np.sqrt(np.average((r - fitted) ** 2, weights=w)))
        lam = n_issuer / (n_issuer + self.k_)
        issuer_term = lam * r_issuer + lam_b * slope * delta_t

        # Comparables hors émetteur : filtres durs relâchés si trop peu de candidats.
        base = keep & ~same_issuer
        same_ccy = a["currency"] == t["currency"]
        same_sen = a["seniority"] == t["seniority"]
        same_cpn = a["coupon_type"] == t["coupon_type"]
        masks = [base & same_ccy & same_sen & same_cpn, base & same_ccy & same_sen, base & same_ccy, base]
        level = next((i for i, m in enumerate(masks) if m.sum() >= cfg["min_candidates"]), len(masks) - 1)
        idx = np.flatnonzero(masks[level])
        # Écarts normalisés entre 0 et 1, dans l'ordre de WEIGHT_KEYS.
        components = np.column_stack([
            np.minimum(np.abs(a["notch"][idx] - t["notch"]) / 6.0, 1.0),
            np.minimum(np.abs(a["log_t"][idx] - t["log_t"]) / 1.5, 1.0),
            (a["sector"][idx] != t["sector"]).astype(float),
            (a["country"][idx] != t["country"]).astype(float),
            np.minimum(np.abs(a["log_amt"][idx] - t["log_amt"]) / 2.0, 1.0),
        ]) if len(idx) else np.zeros((0, len(WEIGHT_KEYS)))

        near = (issuer_mask | base) & (np.abs(a["notch"] - t["notch"]) <= 1)
        tier, reason, seen = classify(
            t, a["currency"][issuer_mask], a["seniority"][issuer_mask],
            a["maturity_years"][issuer_mask], self.settings,
        )
        return {
            "peer_log": peer_log, "lam": lam, "r_issuer": r_issuer, "n_issuer": int(offset_mask.sum()),
            "issuer_term": issuer_term, "issuer_slope": slope, "lam_slope": lam_b, "delta_t": delta_t,
            "issuer_disp": issuer_disp,
            "idx": idx, "components": components, "filter_level": level,
            "issuer_mask": issuer_mask, "near": near,
            "segment_count": int((near & (a["sector"] == t["sector"])).sum()),
            "tier": tier, "tier_reason": reason, "issuer_seen": seen,
            "key": tier if seen else UNSEEN_KEY,
        }

    def _core(self, t: dict, exclude_isin=None, exclude_issuer: bool = False,
              peer_log: float | None = None) -> dict:
        """Modèle A complet pour une cible."""
        core = self.candidates(t, exclude_isin, exclude_issuer, peer_log)
        a, params, idx = self.a_, self.params_, core["idx"]
        r_peers, n_eff, mean_distance, peer_disp = 0.0, 0.0, 1.0, float("nan")
        top, weights, distances = idx[:0], np.array([]), np.array([])
        if len(idx):
            d = core["components"] @ np.array([params["weights"][k] for k in WEIGHT_KEYS])
            order = np.argsort(d, kind="stable")[: params["top_k"]]
            top, distances = idx[order], d[order]
            weights = np.exp(-((distances / params["bandwidth"]) ** 2)) * a["obs_weight"][top]
            total = weights.sum()
            if total > 0:
                r_peers = float((weights * a["resid"][top]).sum() / total)
                # Désaccord des comparables entre eux, une fois ramenés au profil de la cible.
                peer_disp = float(np.sqrt((weights * (a["resid"][top] - r_peers) ** 2).sum() / total))
                n_eff = float(total**2 / (weights**2).sum())
                mean_distance = float((weights * distances).sum() / total)
        core.update({
            "log_a": core["peer_log"] + core["issuer_term"] + (1 - core["lam"]) * r_peers,
            "r_peers": r_peers, "n_eff": n_eff, "mean_distance": mean_distance, "peer_disp": peer_disp,
            "top": top, "weights": weights, "distances": distances,
        })
        return core

    def _ml_logs(self, rows: list[dict], cores: list[dict]) -> tuple[list, list]:
        """Prédictions en log des modèles B et C (None si indisponibles)."""
        none = [None] * len(rows)
        if self.ml_ is None or not rows:
            return none, none
        frame = pd.DataFrame(rows)
        log_b = self.ml_.predict_log(frame).tolist()
        if self.corrector_ is None:
            return log_b, none
        correction = self.corrector_.predict(frame, cores)
        return log_b, [core["log_a"] + float(c) for core, c in zip(cores, correction)]

    # ----------------------------------------------------------- calibration
    def make_folds(self, rng) -> dict[str, np.ndarray]:
        """Deux schémas de validation, pour deux situations réelles :

        - « vu » : plis aléatoires, l'émetteur reste en général dans l'apprentissage ;
        - « non vu » : plis groupés par émetteur, l'émetteur est entièrement retiré.
        """
        n, k = len(self.df_), self.settings["calibration"]["n_folds"]
        issuers = self.a_["issuer"]
        unique = np.unique(issuers)
        issuer_fold = dict(zip(unique, rng.permutation(len(unique)) % k))
        return {
            "vu": rng.permutation(n) % k,
            "non_vu": np.array([issuer_fold[i] for i in issuers]),
        }

    def _calibrate(self) -> dict:
        cal = self.settings["calibration"]
        rng = np.random.default_rng(cal["seed"])
        records = []
        for scheme, folds in self.make_folds(rng).items():
            for fold in np.unique(folds):
                test_ids = np.flatnonzero(folds == fold)
                sub = self.spawn()
                sub.fit_base(self.df_[folds != fold])
                sub.fit_ml()
                rows = self.df_.iloc[test_ids].to_dict("records")
                peer_logs = sub.peer_.predict_log(self.df_.iloc[test_ids])
                cores = [sub._core(row, peer_log=float(pl)) for row, pl in zip(rows, peer_logs)]
                logs_b, logs_c = sub._ml_logs(rows, cores)
                for row_id, row, core, log_b, log_c in zip(test_ids, rows, cores, logs_b, logs_c):
                    anchor = sub.subordination_anchor(row)
                    records.append({
                        "log_s": core["log_a"] if anchor is None else anchor["log"],
                        "row_id": int(row_id), "scheme": scheme, "key": core["key"],
                        "y_log": math.log(row["z_spread"]), "peer_log": core["peer_log"],
                        "log_a": core["log_a"], "log_b": log_b, "log_c": log_c,
                        **error_features(row, core, log_b, log_c, self._dispersion_args()),
                    })
        oof = pd.DataFrame(records)
        # Le schéma « non vu » ne sert qu'au cas de l'émetteur inconnu, et réciproquement.
        oof = oof[(oof["key"] == UNSEEN_KEY) == (oof["scheme"] == "non_vu")]
        # Les obligations ayant servi à régler les comparables sont écartées.
        if self.tuning_ is not None:
            oof = oof[~oof["row_id"].isin(self.tuning_["tune_row_ids"])]
        oof = oof.reset_index(drop=True)

        available = [m for m, column in MODEL_COLUMNS.items() if oof[column].notna().all()]
        if not cal.get("peer_curve_candidate", True):
            available.remove("P")
        champion = {}
        oof["err"] = (oof["log_a"] - oof["y_log"]).abs()
        for key, group in oof.groupby("key"):
            if len(group) < cal["min_per_key"]:
                continue
            medians = {
                m: float((group[MODEL_COLUMNS[m]].astype(float) - group["y_log"]).abs().median())
                for m in available
            }
            # Le modèle A reste champion sauf gain net (marge) d'un autre modèle.
            best = min(medians, key=medians.get)
            beats = medians[best] < (1 - cal["champion_margin"]) * medians["A"]
            champion[key] = best if beats else "A"
            column = MODEL_COLUMNS[champion[key]]
            oof.loc[group.index, "err"] = (group[column].astype(float) - group["y_log"]).abs()

        # Modèle de difficulté, prédit hors échantillon (plis par obligation) pour
        # ne pas flatter les scores de non-conformité.
        x, err = oof[ERROR_FEATURES].to_numpy(dtype=float), oof["err"].to_numpy(dtype=float)
        bond_fold = rng.permutation(len(self.df_))[oof["row_id"].to_numpy()] % 4
        sigma = np.empty(len(oof))
        for fold in range(4):
            held = bond_fold == fold
            sigma[held] = ErrorModel(self.use_ml).fit(x[~held], err[~held]).predict(x[held])
        oof["sigma"] = sigma
        scores = err / sigma if cal["adaptive"] else err
        quantile = {GLOBAL_KEY: conformal_quantile(scores, cal["alpha"])}
        for key, group in oof.groupby("key"):
            if len(group) >= cal["min_per_key"]:
                quantile[key] = conformal_quantile(scores[group.index], cal["alpha"])
        return {
            "alpha": cal["alpha"], "adaptive": cal["adaptive"], "champion": champion,
            "quantile": quantile, "oof": oof, "available_models": available,
            "error_model": ErrorModel(self.use_ml).fit(x, err), "sigma_reference": np.sort(sigma),
        }

    def subordination_anchor(self, t: dict, exclude_isin=None) -> dict | None:
        """Ancrage de la dette subordonnée sur la dette senior de l'émetteur (None si sans objet)."""
        if self.subordination_ is None:
            return None
        shrink = self.settings.get("subordination", {}).get("shrink", True)
        return self.subordination_.anchor(self, t, exclude_isin=exclude_isin, shrink=shrink)

    def _waterfall(self, t: dict, core: dict, champion: str, anchor: dict | None, final_log: float) -> list[dict]:
        """Cascade explicative adaptée au modèle retenu."""
        lam = core["lam"]
        if champion == "S" and anchor is not None:
            issuer_term = anchor["weight"] * anchor["senior_level_log"]
            issuer_label = (f"Niveau de la dette senior de l'émetteur ({anchor['senior_bonds']} obligation(s), "
                            f"retenu à {100 * anchor['weight']:.0f} %)")
            peers_term = anchor["premium_log"] - anchor["generic_premium_log"]
            peers_label = "Prime de subordination mesurée chez les émetteurs, au lieu de la prime moyenne"
        elif champion == "P":
            issuer_term, issuer_label = 0.0, None
            peers_term, peers_label = 0.0, None
        else:
            issuer_term = core["issuer_term"]
            issuer_label = None if core["n_issuer"] == 0 else (
                f"Écart émetteur ({core['n_issuer']} obligation(s), retenu à {100 * lam:.0f} %"
                + (", pente de maturité comprise)" if core["lam_slope"] > 0.05 and abs(core["delta_t"]) > 0.05 else ")"))
            peers_term, peers_label = (1 - lam) * core["r_peers"], "Comparables hors émetteur"
        return build_waterfall(self.peer_.mean_log_, self.peer_.group_contributions(t), issuer_term, issuer_label,
                               peers_term, final_log, peers_label=peers_label)

    def _dispersion_args(self) -> dict | None:
        """Indicateurs de dispersion pour le modèle de difficulté (None : désactivés)."""
        if not self.settings["confidence"].get("dispersion_features", False):
            return None
        return {"noise_sd": self.noise_sd_}

    def backtest_summary(self) -> list[dict]:
        """Erreur hors échantillon par palier, en points de base."""
        if self.calibration_ is None:
            raise RuntimeError("Estimateur non calibré : appeler fit(calibrate=True).")
        oof = self.calibration_["oof"]
        truth = np.exp(oof["y_log"])
        names = {"courbe_pairs": "peer_log", "A_comparables": "log_a",
                 "B_catboost": "log_b", "C_comparables_corriges": "log_c", "S_ancrage_senior": "log_s"}
        rows = []
        for key, group in oof.groupby("key"):
            row = {"palier": key, "n": len(group), "champion": self.calibration_["champion"].get(key, "A")}
            for name, column in names.items():
                if group[column].isna().any():
                    continue
                error = (np.exp(group[column].astype(float)) - truth[group.index]).abs()
                row[f"mae_bp_{name}"] = round(float(error.mean()), 1)
                row[f"mediane_abs_bp_{name}"] = round(float(error.median()), 1)
            rows.append(row)
        return rows

    def calibration_report(self) -> dict:
        """Ce qui a été appris sur les données plutôt que fixé à la main."""
        tuning = None
        if self.tuning_ is not None:
            tuning = {k: v for k, v in self.tuning_.items() if k != "tune_row_ids"}
        return _jsonable({
            "issuer_shrinkage": self.shrinkage_,
            "issuer_slope": self.slope_,
            "peer_curve_enriched": self.settings["peer_curve"].get("enriched", False),
            "dispersion_features": self.settings["confidence"].get("dispersion_features", False),
            "comparables_params": self.params_,
            "comparables_tuning": tuning,
            "champion_by_tier": (self.calibration_ or {}).get("champion"),
            "adaptive_interval": (self.calibration_ or {}).get("adaptive"),
        })

    # ------------------------------------------------------------ estimation
    def resolve(self, bond: str | dict) -> tuple[dict, str | None, float | None]:
        """Renvoie (cible préparée, ISIN à exclure, spread observé éventuel)."""
        if isinstance(bond, str):
            match = self.df_[self.df_["isin"] == bond]
            if match.empty:
                raise KeyError(f"ISIN {bond} absent de l'univers : fournir les caractéristiques.")
            row = match.iloc[0].to_dict()
            return row, bond, float(row["z_spread"])
        target = prepare_target(bond)
        if bond.get("amount_outstanding") in (None, "") or pd.isna(bond.get("amount_outstanding")):
            # Taille inconnue : on prend la taille médiane de l'univers, pas une taille nulle.
            target["log_amt"] = float(np.median(self.a_["log_amt"]))
        return target, target.get("isin"), None

    def estimate(self, bond: str | dict) -> dict:
        """Estime le Z-Spread juste d'une obligation (ISIN de l'univers ou caractéristiques).

        Une obligation déjà présente dans l'univers est exclue de ses propres comparables.
        """
        t, exclude, observed = self.resolve(bond)
        core = self._core(t, exclude_isin=exclude)
        (log_b,), (log_c,) = self._ml_logs([t], [core])
        anchor = self.subordination_anchor(t, exclude_isin=exclude)
        logs = {"A": core["log_a"], "B": log_b, "C": log_c, "P": core["peer_log"],
                "S": None if anchor is None else anchor["log"]}
        a, cal = self.a_, self.calibration_

        champion = (cal or {}).get("champion", {}).get(core["key"], "A")
        if logs[champion] is None:
            champion = "A"
        final_log = logs[champion]
        estimate_bp = math.exp(final_log)
        gap = None if log_b is None else core["log_a"] - log_b

        caveats = list(STANDING_CAVEATS)
        interval, confidence = None, {"score": None, "grade": None}
        if cal is not None:
            features = error_features(t, core, log_b, log_c, self._dispersion_args())
            sigma = float(cal["error_model"].predict([[features[f] for f in ERROR_FEATURES]])[0])
            q = cal["quantile"].get(core["key"])
            if q is None:
                q = cal["quantile"][GLOBAL_KEY]
                caveats.append("Palier trop peu représenté dans le backtest : intervalle calibré tous paliers confondus.")
            half = q * sigma if cal["adaptive"] else q
            interval = [round(math.exp(final_log - half), 1), round(math.exp(final_log + half), 1)]
            confidence = confidence_from_difficulty(
                sigma, cal["sigma_reference"], estimate_bp,
                {
                    "effective_comparables": round(core["n_eff"], 1),
                    "mean_distance": round(core["mean_distance"], 3),
                    "issuer_weight": round(core["lam"], 3),
                    "model_gap_pct": None if gap is None else round(100 * abs(gap), 1),
                    "segment_count": core["segment_count"],
                },
                self.settings["confidence"]["grades"],
            )
        if core["filter_level"] >= 2:
            caveats.append(f"Filtres durs relâchés ({FILTER_LEVELS[core['filter_level']]}).")

        near = core["near"]
        bounds = tuple(np.percentile(a["z_spread"][near], [1, 99])) if near.sum() >= 10 else None
        senior_check = None
        if anchor is not None:
            raw = np.isin(a["isin"], anchor["senior_isins"])
            senior_check = {"equivalent_bp": math.exp(anchor["senior_equivalent_log"]),
                            "raw_median_bp": float(np.median(a["z_spread"][raw])), "n": anchor["senior_bonds"]}

        comparables = [
            {
                "isin": a["isin"][i], "issuer": a["issuer"][i], "rating": self.df_["rating"].iat[i],
                "seniority": a["seniority"][i], "sector": a["sector"][i],
                "maturity_years": round(float(a["maturity_years"][i]), 2),
                "z_spread_bp": round(float(a["z_spread"][i]), 1),
                # Spread du comparable translaté aux caractéristiques de la cible.
                "adjusted_z_spread_bp": round(math.exp(core["peer_log"] + a["resid"][i]), 1),
                "distance": round(float(d), 3), "weight": round(float(w), 3),
            }
            for i, d, w in zip(core["top"], core["distances"], core["weights"])
        ]
        lam = core["lam"]
        result = {
            "bond": {c: t.get(c) for c in DESCRIPTIVE_COLUMNS},
            "z_spread_bp": round(estimate_bp, 1),
            "interval_bp": interval,
            "interval_level": 1 - self.settings["calibration"]["alpha"],
            "observed_z_spread_bp": observed,
            "tier": core["tier"], "tier_reason": core["tier_reason"], "issuer_seen": core["issuer_seen"],
            "champion": champion,
            "models": {
                "courbe_pairs_bp": round(math.exp(core["peer_log"]), 1),
                "A_comparables_bp": round(math.exp(core["log_a"]), 1),
                "B_catboost_bp": None if log_b is None else round(math.exp(log_b), 1),
                "C_comparables_corriges_bp": None if log_c is None else round(math.exp(log_c), 1),
                "S_ancrage_senior_bp": None if anchor is None else round(math.exp(anchor["log"]), 1),
                "top_down_bp": None,
            },
            "confidence": confidence,
            "issuer_offset": {
                "n_bonds": core["n_issuer"], "mean_residual_log": round(core["r_issuer"], 4),
                "lambda": round(lam, 3), "k": round(self.k_, 3),
                "slope_log_per_log_year": round(core["issuer_slope"], 4), "slope_weight": round(core["lam_slope"], 3),
                "slope_effect_log": round(core["lam_slope"] * core["issuer_slope"] * core["delta_t"], 4),
            },
            "filter_level": FILTER_LEVELS[core["filter_level"]],
            "comparables": comparables,
            "waterfall": self._waterfall(t, core, champion, anchor, final_log),
            "controls": run_controls(
                estimate_bp, bounds, senior_check, gap, core["n_eff"], lam, self.settings
            ),
            "caveats": caveats,
            "model_version": self.model_version_,
        }
        if anchor is not None:
            result["subordination"] = {
                "senior_bonds": anchor["senior_bonds"], "senior_isins": anchor["senior_isins"],
                "senior_equivalent_bp": round(math.exp(anchor["senior_equivalent_log"]), 1),
                "senior_weight": round(anchor["weight"], 3),
                "premium_pct": round(100 * (math.exp(anchor["premium_log"]) - 1), 1),
                "generic_premium_pct": round(100 * (math.exp(anchor["generic_premium_log"]) - 1), 1),
                "premium_source": anchor["premium_source"],
                "anchor_bp": round(math.exp(anchor["log"]), 1),
            }
        # Lecture descendante : contrôle lisible, pas le chiffre retenu.
        reader = self.settings.get("top_down", {}).get("reader", "Entonnoir avec maturité")
        reading = read_control(self, t, reader, exclude_isin=exclude, explain=True)
        reading.pop("log")
        reading["gap_pct"] = round(100 * (reading["z_spread_bp"] / estimate_bp - 1), 1)
        result["top_down"] = reading
        result["models"]["top_down_bp"] = reading["z_spread_bp"]
        limit = 100 * self.settings["top_down"]["max_gap"] if "top_down" in self.settings else 25.0
        result["controls"].append({
            "control": "Accord avec la lecture descendante",
            "status": "ok" if abs(reading["gap_pct"]) <= limit else "alerte",
            "detail": f"Lecture descendante {reading['z_spread_bp']:.0f} pb ({reading['level_name']}) ; "
                      f"écart {reading['gap_pct']:+.0f} % avec l'estimation.",
        })
        if self.ml_ is not None:
            result["shap_log_catboost"] = {k: round(v, 4) for k, v in self.ml_.shap_log(t).items()}
        if self.corrector_ is not None:
            result["shap_log_correction"] = {
                k: round(v, 4) for k, v in self.corrector_.shap_log(t, core).items()
            }
        result["narrative"] = build_narrative(result)
        return _jsonable(result)


def _jsonable(value):
    """Convertit les types numpy et les NaN pour une sérialisation JSON propre."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value
