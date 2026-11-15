"""Chargement des paramètres : valeurs par défaut du code, surchargées par YAML."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "tiers": {
        "min_issuer_bonds_simple": 3,
        "complex_seniorities": ["TIER2", "AT1", "SUBORDINATED", "HYBRID"],
    },
    "comparables": {
        "top_k": 15,
        "min_candidates": 5,
        "bandwidth": 0.25,
        "issuer_shrinkage_k": "auto",
        "tune": True,
        "tune_trials": 200,
        "tune_share": 0.10,
        "tune_min_weight": 0.05,
        "issuer_slope": True,
        "weights": {"notch": 0.35, "maturity": 0.25, "sector": 0.25, "country": 0.10, "size": 0.05},
    },
    "peer_curve": {"ridge": 0.01, "enriched": True},
    "ml": {
        "iterations": 300, "depth": 5, "learning_rate": 0.08,
        "corrector": True, "corrector_iterations": 200, "corrector_depth": 4,
    },
    "calibration": {"alpha": 0.10, "n_folds": 5, "min_per_key": 50, "seed": 7, "adaptive": True,
        "champion_margin": 0.05,
    },
    "confidence": {"grades": {"A": 75, "B": 50}, "dispersion_features": True},
    "controls": {"max_model_gap": 0.25, "min_effective_comparables": 3.0},
    "feedback": {"weights": {"trade": 1.0, "opinion": 0.3}},
    "analysis": {"clustering_reading_k": 4},
}


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_settings(path: str | Path | None = None) -> dict:
    """Renvoie les paramètres ; `path` pointe vers un settings.yaml optionnel."""
    settings = copy.deepcopy(DEFAULTS)
    if path is not None and Path(path).exists():
        with open(path, encoding="utf-8") as handle:
            _merge(settings, yaml.safe_load(handle) or {})
    return settings
