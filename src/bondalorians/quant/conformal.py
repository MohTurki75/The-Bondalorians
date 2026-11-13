"""Prédiction conforme : quantile corrigé pour échantillon fini."""

from __future__ import annotations

import math

import numpy as np


def conformal_quantile(abs_errors: np.ndarray, alpha: float) -> float:
    """Quantile d'ordre ceil((n+1)(1-alpha))/n des erreurs absolues hors échantillon.

    Garantit une couverture marginale d'au moins 1-alpha si les erreurs de
    calibration et la nouvelle observation sont échangeables.
    """
    errors = np.sort(np.asarray(abs_errors, dtype=float))
    n = len(errors)
    if n == 0:
        raise ValueError("Aucune erreur de calibration.")
    rank = math.ceil((n + 1) * (1 - alpha))
    return float(errors[min(rank, n) - 1])
