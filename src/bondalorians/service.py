"""Façade applicative : la seule porte d'entrée du serveur MCP et de toute interface.

Fonctionne sans MCP ni LLM, pour que la démo survive à une panne de l'un ou l'autre.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from bondalorians.config import load_settings
from bondalorians.data import repository
from bondalorians.data.schema import DESCRIPTIVE_COLUMNS
from bondalorians.data.stores import JsonlStore, PostgresStore
from bondalorians.learning.feedback import augment_universe, validate_feedback
from bondalorians.quant.estimator import ZSpreadEstimator, _jsonable

SEARCHABLE = ["issuer", "sector", "country", "currency", "rating", "seniority", "coupon_type"]


class Service:
    def __init__(self, universe: pd.DataFrame, settings: dict | None = None,
                 run_store=None, feedback_store=None, use_ml: bool = True):
        self.settings = settings or load_settings()
        self.base_universe = universe
        self.run_store = run_store or JsonlStore("runs/runs.jsonl")
        self.feedback_store = feedback_store or JsonlStore("runs/feedback.jsonl")
        self.estimator = ZSpreadEstimator(self.settings, use_ml=use_ml)
        self.estimator.fit(self._current_universe(), calibrate=True)

    @classmethod
    def from_env(cls) -> "Service":
        """Construit le service à partir des variables d'environnement.

        BONDALORIANS_DSN (PostgreSQL) ou BONDALORIANS_CSV (hors ligne) ;
        BONDALORIANS_CONFIG_DIR (défaut : ./config) ; BONDALORIANS_USE_ML=0 pour couper CatBoost.
        """
        config_dir = Path(os.environ.get("BONDALORIANS_CONFIG_DIR", "config"))
        settings = load_settings(config_dir / "settings.yaml")
        use_ml = os.environ.get("BONDALORIANS_USE_ML", "1") != "0"
        dsn = os.environ.get("BONDALORIANS_DSN")
        if dsn:
            schema_map = repository.load_schema_map(config_dir / "schema_map.yaml")
            universe = repository.load_universe_postgres(dsn, schema_map)
            return cls(
                universe, settings, use_ml=use_ml,
                run_store=PostgresStore(dsn, "app.runs", "run_id"),
                feedback_store=PostgresStore(dsn, "app.feedback", "feedback_id"),
            )
        csv_path = os.environ.get("BONDALORIANS_CSV")
        if not csv_path:
            raise RuntimeError("Définir BONDALORIANS_DSN ou BONDALORIANS_CSV.")
        return cls(repository.load_universe_csv(csv_path), settings, use_ml=use_ml)

    def _current_universe(self) -> pd.DataFrame:
        return augment_universe(self.base_universe, self.feedback_store.all(), self.settings)

    # ------------------------------------------------------------- lecture
    def describe_universe(self) -> dict:
        df = self.estimator.df_
        real = df[~df["isin"].astype(str).str.startswith("FB_")]
        return _jsonable({
            "n_bonds": len(real), "n_issuers": int(real["issuer"].nunique()),
            "n_feedback_observations": len(df) - len(real),
            "rows_dropped_as_unusable": self.estimator.dropped_,
            "z_spread_bp": real["z_spread"].describe().round(1).to_dict(),
            "by_currency": real["currency"].value_counts().to_dict(),
            "by_seniority": real["seniority"].value_counts().to_dict(),
            "by_sector": real["sector"].value_counts().to_dict(),
            "by_rating": real["rating"].value_counts().to_dict(),
            "model_version": self.estimator.model_version_,
        })

    def get_bond(self, isin: str) -> dict:
        df = self.estimator.df_
        match = df[df["isin"] == isin]
        if match.empty:
            raise KeyError(f"ISIN {isin} absent de l'univers.")
        return _jsonable(match.iloc[0][DESCRIPTIVE_COLUMNS + ["z_spread"]].to_dict())

    def search_bonds(self, filters: dict | None = None, limit: int = 25) -> list[dict]:
        """Filtres d'égalité sur les colonnes descriptives, plus min/max de maturité."""
        df = self.estimator.df_
        filters = dict(filters or {})
        low, high = filters.pop("min_maturity", None), filters.pop("max_maturity", None)
        unknown = set(filters) - set(SEARCHABLE)
        if unknown:
            raise ValueError(f"Filtres inconnus : {sorted(unknown)} (autorisés : {SEARCHABLE}).")
        for column, value in filters.items():
            df = df[df[column].astype(str).str.upper() == str(value).upper()]
        if low is not None:
            df = df[df["maturity_years"] >= low]
        if high is not None:
            df = df[df["maturity_years"] <= high]
        columns = DESCRIPTIVE_COLUMNS + ["z_spread"]
        return _jsonable(df[columns].head(min(limit, 200)).to_dict("records"))

    # ---------------------------------------------------------- estimation
    def classify_complexity(self, bond: str | dict) -> dict:
        target, exclude, _ = self.estimator.resolve(bond)
        core = self.estimator._core(target, exclude_isin=exclude)
        return {"tier": core["tier"], "reason": core["tier_reason"], "issuer_seen": core["issuer_seen"]}

    def estimate(self, bond: str | dict) -> dict:
        result = self.estimator.estimate(bond)
        result["run_id"] = self.run_store.save({"input": bond, "result": result}, prefix="run")
        return result

    def find_comparables(self, bond: str | dict) -> dict:
        result = self.estimator.estimate(bond)
        return {k: result[k] for k in ("filter_level", "issuer_offset", "comparables")}

    def get_run(self, run_id: str) -> dict:
        record = self.run_store.get(run_id)
        if record is None:
            raise KeyError(f"Estimation {run_id} introuvable.")
        return record

    def validate_estimate(self, run_id: str) -> dict:
        result = self.get_run(run_id)["payload"]["result"]
        alerts = [c for c in result["controls"] if c["status"] != "ok"]
        return {"run_id": run_id, "passed": not alerts, "controls": result["controls"]}

    def explain_estimate(self, run_id: str) -> dict:
        result = self.get_run(run_id)["payload"]["result"]
        keys = ["z_spread_bp", "interval_bp", "interval_level", "tier", "tier_reason", "champion",
                "models", "confidence", "issuer_offset", "filter_level", "waterfall",
                "comparables", "controls", "caveats", "narrative", "model_version"]
        out = {k: result[k] for k in keys}
        if "shap_log_catboost" in result:
            out["shap_log_catboost"] = result["shap_log_catboost"]
        return out

    def backtest_summary(self) -> list[dict]:
        return self.estimator.backtest_summary()

    def calibration_report(self) -> dict:
        return self.estimator.calibration_report()

    # ------------------------------------------------------------ feedback
    def submit_feedback(self, run_id: str, kind: str, z_spread_bp: float,
                        reason: str = "", author: str = "") -> dict:
        """Enregistre un retour et l'intègre aussitôt comme observation pondérée.

        Le réajustement est rapide : courbe des pairs, résidus et k sont
        recalculés ; les modèles CatBoost, les paramètres de comparables et la
        calibration restent ceux du dernier ajustement complet.
        """
        validate_feedback(kind, z_spread_bp)
        result = self.get_run(run_id)["payload"]["result"]
        feedback_id = self.feedback_store.save({
            "run_id": run_id, "kind": kind, "z_spread_bp": float(z_spread_bp),
            "reason": reason, "author": author, "bond": result["bond"],
            "estimate_bp": result["z_spread_bp"],
        }, prefix="fb")
        self.estimator.refit(self._current_universe())
        return {
            "feedback_id": feedback_id,
            "weight": self.settings["feedback"]["weights"][kind],
            "model_version": self.estimator.model_version_,
        }
