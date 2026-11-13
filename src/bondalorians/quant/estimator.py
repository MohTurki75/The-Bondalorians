"""Estimateur du Z-Spread juste.

Modèle A (transparent) : en log,
    spread = courbe des pairs + lambda * écart émetteur + (1 - lambda) * écart des comparables
où lambda = n / (n + k) rétrécit l'écart émetteur quand l'émetteur a peu d'obligations.
Modèle B (challenger) : CatBoost. Le backtest désigne le champion par palier
et fournit les quantiles de l'intervalle conforme.
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
from bondalorians.quant.confidence import confidence_score
from bondalorians.quant.conformal import conformal_quantile
from bondalorians.quant.gbm import SpreadGBM, catboost_available
from bondalorians.quant.peer_curve import PeerCurve

UNSEEN_KEY = "emetteur_inconnu"
FILTER_LEVELS = [
    "devise, séniorité et type de coupon",
    "devise et séniorité",
    "devise seule",
    "aucun filtre dur",
]
_ARRAYS_OBJECT = ["isin", "issuer", "currency", "seniority", "coupon_type", "sector", "country"]
_ARRAYS_FLOAT = ["notch", "log_t", "maturity_years", "log_amt", "sen_rank", "obs_weight", "z_spread"]


class ZSpreadEstimator:
    def __init__(self, settings: dict | None = None, use_ml: bool = True):
        self.settings = settings or load_settings()
        self.use_ml = use_ml and catboost_available()
        self.calibration_: dict | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, universe: pd.DataFrame, calibrate: bool = True) -> "ZSpreadEstimator":
        df = prepare(universe, require_target=True)
        self.dropped_ = df.attrs.get("dropped", 0)
        self._fit_prepared(df)
        if calibrate:
            self.calibration_ = self._calibrate()
        return self

    def _fit_prepared(self, df: pd.DataFrame) -> None:
        self.df_ = df.reset_index(drop=True)
        self.peer_ = PeerCurve(self.settings["peer_curve"]["ridge"]).fit(self.df_)
        self.a_ = {c: self.df_[c].to_numpy(dtype=object) for c in _ARRAYS_OBJECT}
        self.a_.update({c: self.df_[c].to_numpy(dtype=float) for c in _ARRAYS_FLOAT})
        self.a_["resid"] = np.log(self.a_["z_spread"]) - self.peer_.predict_log(self.df_)
        self.ml_ = None
        if self.use_ml:
            ml = self.settings["ml"]
            self.ml_ = SpreadGBM(ml["iterations"], ml["depth"], ml["learning_rate"]).fit(self.df_)
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(self.df_[["issuer", "maturity_years", "z_spread"]]).to_numpy().tobytes()
        ).hexdigest()[:10]
        self.model_version_ = f"{__version__}-{len(self.df_)}-{digest}"

    # ----------------------------------------------------------------- cœur
    def _core(self, t: dict, exclude_isin=None, peer_log: float | None = None,
              log_b: float | None = None) -> dict:
        a, cfg = self.a_, self.settings["comparables"]
        keep = np.ones(len(self.df_), dtype=bool)
        if exclude_isin is not None:
            keep &= a["isin"] != exclude_isin
        if peer_log is None:
            peer_log = float(self.peer_.predict_log(pd.DataFrame([t]))[0])

        # Écart émetteur, rétréci selon le nombre d'obligations.
        issuer_mask = keep & (a["issuer"] == t["issuer"])
        offset_mask = issuer_mask & (a["currency"] == t["currency"])
        if not offset_mask.any():
            offset_mask = issuer_mask
        n_issuer = float(a["obs_weight"][offset_mask].sum())
        r_issuer = 0.0
        if n_issuer > 0:
            r_issuer = float(np.average(a["resid"][offset_mask], weights=a["obs_weight"][offset_mask]))
        lam = n_issuer / (n_issuer + cfg["issuer_shrinkage_k"])

        # Comparables hors émetteur : filtres durs relâchés si trop peu de candidats.
        base = keep & (a["issuer"] != t["issuer"])
        same_ccy = a["currency"] == t["currency"]
        same_sen = a["seniority"] == t["seniority"]
        same_cpn = a["coupon_type"] == t["coupon_type"]
        masks = [base & same_ccy & same_sen & same_cpn, base & same_ccy & same_sen, base & same_ccy, base]
        level = next((i for i, m in enumerate(masks) if m.sum() >= cfg["min_candidates"]), len(masks) - 1)
        idx = np.flatnonzero(masks[level])

        r_peers, n_eff, mean_distance = 0.0, 0.0, 1.0
        top, weights, distances = idx[:0], np.array([]), np.array([])
        if len(idx):
            w = cfg["weights"]
            d = (
                w["notch"] * np.minimum(np.abs(a["notch"][idx] - t["notch"]) / 6.0, 1.0)
                + w["maturity"] * np.minimum(np.abs(a["log_t"][idx] - t["log_t"]) / 1.5, 1.0)
                + w["sector"] * (a["sector"][idx] != t["sector"])
                + w["country"] * (a["country"][idx] != t["country"])
                + w["size"] * np.minimum(np.abs(a["log_amt"][idx] - t["log_amt"]) / 2.0, 1.0)
            ).astype(float)
            order = np.argsort(d, kind="stable")[: cfg["top_k"]]
            top, distances = idx[order], d[order]
            weights = np.exp(-((distances / cfg["bandwidth"]) ** 2)) * a["obs_weight"][top]
            total = weights.sum()
            if total > 0:
                r_peers = float((weights * a["resid"][top]).sum() / total)
                n_eff = float(total**2 / (weights**2).sum())
                mean_distance = float((weights * distances).sum() / total)

        log_a = peer_log + lam * r_issuer + (1 - lam) * r_peers
        if log_b is None and self.ml_ is not None:
            log_b = float(self.ml_.predict_log(pd.DataFrame([t]))[0])
        tier, reason, seen = classify(
            t, a["currency"][issuer_mask], a["seniority"][issuer_mask],
            a["maturity_years"][issuer_mask], self.settings,
        )
        return {
            "peer_log": peer_log, "log_a": log_a, "log_b": log_b,
            "lambda": lam, "r_issuer": r_issuer, "n_issuer": int(offset_mask.sum()),
            "r_peers": r_peers, "n_eff": n_eff, "mean_distance": mean_distance,
            "top": top, "weights": weights, "distances": distances, "filter_level": level,
            "tier": tier, "tier_reason": reason, "issuer_seen": seen,
            "key": tier if seen else UNSEEN_KEY, "issuer_mask": issuer_mask,
        }

    # ----------------------------------------------------------- calibration
    def _calibrate(self) -> dict:
        """Validation croisée sous deux schémas, pour deux situations réelles :

        - « vu » : plis aléatoires, l'émetteur reste en général dans l'apprentissage ;
        - « non vu » : plis groupés par émetteur, l'émetteur est entièrement retiré.
        """
        cal = self.settings["calibration"]
        rng = np.random.default_rng(cal["seed"])
        n, k = len(self.df_), cal["n_folds"]
        issuers = self.df_["issuer"].to_numpy(dtype=object)
        unique = np.unique(issuers)
        issuer_fold = dict(zip(unique, rng.permutation(len(unique)) % k))
        schemes = {
            "vu": rng.permutation(n) % k,
            "non_vu": np.array([issuer_fold[i] for i in issuers]),
        }
        records = []
        for scheme, folds in schemes.items():
            for fold in range(k):
                train, test = self.df_[folds != fold], self.df_[folds == fold]
                if test.empty or train.empty:
                    continue
                sub = ZSpreadEstimator(self.settings, use_ml=self.use_ml)
                sub._fit_prepared(train)
                peer_logs = sub.peer_.predict_log(test)
                logs_b = sub.ml_.predict_log(test) if sub.ml_ is not None else [None] * len(test)
                for row, peer_log, log_b in zip(test.to_dict("records"), peer_logs, logs_b):
                    core = sub._core(row, peer_log=float(peer_log), log_b=None if log_b is None else float(log_b))
                    records.append({
                        "scheme": scheme, "key": core["key"], "tier": core["tier"],
                        "y_log": math.log(row["z_spread"]), "peer_log": core["peer_log"],
                        "log_a": core["log_a"], "log_b": core["log_b"],
                    })
        oof = pd.DataFrame(records)
        # Le schéma « non vu » ne sert qu'au cas de l'émetteur inconnu, et réciproquement.
        oof = oof[(oof["key"] == UNSEEN_KEY) == (oof["scheme"] == "non_vu")].reset_index(drop=True)

        champion, quantile = {}, {}
        for key, group in oof.groupby("key"):
            if len(group) < cal["min_per_key"]:
                continue
            err_a = (group["log_a"] - group["y_log"]).abs()
            best, errors = "A", err_a
            if self.ml_ is not None:
                err_b = (group["log_b"].astype(float) - group["y_log"]).abs()
                if err_b.median() < err_a.median():
                    best, errors = "B", err_b
            champion[key] = best
            quantile[key] = conformal_quantile(errors.to_numpy(), cal["alpha"])
        return {"alpha": cal["alpha"], "champion": champion, "quantile": quantile, "oof": oof}

    def backtest_summary(self) -> list[dict]:
        """Erreur hors échantillon par palier, en points de base."""
        if self.calibration_ is None:
            raise RuntimeError("Estimateur non calibré : appeler fit(calibrate=True).")
        oof = self.calibration_["oof"]
        truth = np.exp(oof["y_log"])
        rows = []
        models = {"courbe_pairs": "peer_log", "A_comparables": "log_a"}
        if self.ml_ is not None:
            models["B_catboost"] = "log_b"
        for key, group in oof.groupby("key"):
            row = {"palier": key, "n": len(group), "champion": self.calibration_["champion"].get(key)}
            for name, column in models.items():
                error = np.exp(group[column].astype(float)) - truth[group.index]
                row[f"mae_bp_{name}"] = round(float(error.abs().mean()), 1)
                row[f"mediane_abs_bp_{name}"] = round(float(error.abs().median()), 1)
            rows.append(row)
        return rows

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
        return target, target.get("isin"), None

    def estimate(self, bond: str | dict) -> dict:
        """Estime le Z-Spread juste d'une obligation (ISIN de l'univers ou caractéristiques).

        Une obligation déjà présente dans l'univers est exclue de ses propres comparables.
        """
        t, exclude, observed = self.resolve(bond)
        core = self._core(t, exclude_isin=exclude)
        a, cal = self.a_, self.calibration_

        champion = (cal or {}).get("champion", {}).get(core["key"], "A")
        if champion == "B" and core["log_b"] is None:
            champion = "A"
        final_log = core["log_a"] if champion == "A" else core["log_b"]
        estimate_bp = math.exp(final_log)

        q = (cal or {}).get("quantile", {}).get(core["key"])
        interval = [round(math.exp(final_log - q), 1), round(math.exp(final_log + q), 1)] if q else None
        gap = None if core["log_b"] is None else core["log_a"] - core["log_b"]

        keep = np.ones(len(self.df_), dtype=bool) if exclude is None else a["isin"] != exclude
        near = keep & (np.abs(a["notch"] - t["notch"]) <= 1)
        bounds = tuple(np.percentile(a["z_spread"][near], [1, 99])) if near.sum() >= 10 else None
        segment_count = int((near & (a["sector"] == t["sector"])).sum())
        senior = core["issuer_mask"] & (a["sen_rank"] < t["sen_rank"]) & (a["currency"] == t["currency"])
        senior_ref = float(np.median(a["z_spread"][senior])) if senior.any() else None

        caveats = list(STANDING_CAVEATS)
        if interval is None:
            caveats.append("Intervalle indisponible : palier trop peu représenté dans le backtest.")
        if core["filter_level"] >= 2:
            caveats.append(f"Filtres durs relâchés ({FILTER_LEVELS[core['filter_level']]}).")

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
        lam = core["lambda"]
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
                "B_catboost_bp": None if core["log_b"] is None else round(math.exp(core["log_b"]), 1),
            },
            "confidence": confidence_score(
                q, core["n_eff"], core["mean_distance"], lam, gap, segment_count, self.settings
            ),
            "issuer_offset": {
                "n_bonds": core["n_issuer"], "mean_residual_log": round(core["r_issuer"], 4),
                "lambda": round(lam, 3),
            },
            "filter_level": FILTER_LEVELS[core["filter_level"]],
            "comparables": comparables,
            "waterfall": build_waterfall(
                self.peer_.mean_log_, self.peer_.group_contributions(t),
                lam * core["r_issuer"],
                f"Écart émetteur ({core['n_issuer']} obligation(s), retenu à {100 * lam:.0f} %)",
                (1 - lam) * core["r_peers"], final_log,
            ),
            "controls": run_controls(
                estimate_bp, bounds, senior_ref, gap, core["n_eff"], lam, self.settings
            ),
            "caveats": caveats,
            "model_version": self.model_version_,
        }
        if self.ml_ is not None:
            result["shap_log_catboost"] = {k: round(v, 4) for k, v in self.ml_.shap_log(t).items()}
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
