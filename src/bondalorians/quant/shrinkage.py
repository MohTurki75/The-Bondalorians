"""Rétrécissement de l'écart émetteur : k estimé par les données, sans ML.

Modèle à effet aléatoire sur les résidus de la courbe des pairs :
    résidu(obligation j de l'émetteur i) = alpha_i + bruit_ij
Le meilleur prédicteur de alpha_i est lambda * moyenne des résidus de
l'émetteur, avec lambda = n / (n + k) et k = variance du bruit / variance
entre émetteurs. Les deux variances viennent d'une analyse de variance à un
facteur (estimateur des moments, plan non équilibré).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

K_BOUNDS = (0.05, 50.0)


def estimate_shrinkage_k(residuals: np.ndarray, issuers: np.ndarray) -> dict:
    frame = pd.DataFrame({"r": np.asarray(residuals, dtype=float), "issuer": issuers})
    groups = frame.groupby("issuer")["r"]
    sizes, means = groups.size().to_numpy(dtype=float), groups.mean().to_numpy()
    n_total, n_groups = sizes.sum(), len(sizes)
    if n_groups < 2 or n_total <= n_groups:
        return {"k": 2.0, "noise_sd": None, "issuer_sd": None, "estimated": False}

    within = float(((frame["r"] - groups.transform("mean")) ** 2).sum() / (n_total - n_groups))
    between_ms = float((sizes * (means - frame["r"].mean()) ** 2).sum() / (n_groups - 1))
    n0 = (n_total - (sizes**2).sum() / n_total) / (n_groups - 1)
    issuer_var = max((between_ms - within) / n0, 1e-12)
    k = float(np.clip(within / issuer_var, *K_BOUNDS))
    return {
        "k": k, "noise_sd": round(within**0.5, 4), "issuer_sd": round(issuer_var**0.5, 4),
        "estimated": True,
    }
