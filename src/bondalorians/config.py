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
        "issuer_shrinkage_k": 2.0,
        "weights": {"notch": 0.35, "maturity": 0.25, "sector": 0.25, "country": 0.10, "size": 0.05},
    },
    "peer_curve": {"ridge": 0.01},
    "ml": {"iterations": 300, "depth": 5, "learning_rate": 0.08},
    "calibration": {"alpha": 0.10, "n_folds": 5, "min_per_key": 50, "seed": 7},
    "confidence": {
        "weights": {"interval": 0.40, "comparables": 0.30, "agreement": 0.20, "domain": 0.10},
        "interval_ref": 0.60,
        "agreement_ref": 0.30,
        "domain_ref": 30,
        "grades": {"A": 75, "B": 50},
    },
    "controls": {"max_model_gap": 0.25, "min_effective_comparables": 3.0},
    "feedback": {"weights": {"trade": 1.0, "opinion": 0.3}},
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
