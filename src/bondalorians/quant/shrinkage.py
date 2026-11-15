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


def estimate_slope_shrinkage(residuals: np.ndarray, issuers: np.ndarray, log_t: np.ndarray,
                             noise_var: float | None, min_bonds: int = 3) -> dict:
    """Rétrécissement de la pente de maturité propre à chaque émetteur.

    Pour chaque émetteur, pente b_i des résidus sur le log de la maturité (moindres
    carrés), de variance d'échantillonnage s² / Sxx_i. La variance des vraies pentes
    entre émetteurs vaut, par les moments : tau² = var(b_i) − moyenne(s² / Sxx_i).
    La pente retenue est lambda_b × b_i avec lambda_b = Sxx / (Sxx + k_b),
    k_b = s² / tau² : un émetteur dont les maturités sont proches les unes des
    autres (Sxx petit) voit sa pente fortement réduite.
    """
    frame = pd.DataFrame({"r": residuals, "t": log_t, "issuer": issuers})
    slopes, sxx_list = [], []
    for _, group in frame.groupby("issuer"):
        if len(group) < min_bonds:
            continue
        centered = group["t"] - group["t"].mean()
        sxx = float((centered**2).sum())
        if sxx < 0.05:
            continue
        slopes.append(float((centered * (group["r"] - group["r"].mean())).sum() / sxx))
        sxx_list.append(sxx)
    if len(slopes) < 10 or not noise_var:
        return {"k_b": None, "slope_sd": None, "issuers_used": len(slopes), "estimated": False}
    slopes, sxx_arr = np.array(slopes), np.array(sxx_list)
    tau2 = float(np.var(slopes, ddof=1) - np.mean(noise_var / sxx_arr))
    if tau2 <= 1e-6:
        # Aucune dispersion des pentes au-delà du bruit : pente émetteur neutralisée.
        return {"k_b": float("inf"), "slope_sd": 0.0, "issuers_used": len(slopes), "estimated": True}
    return {"k_b": float(np.clip(noise_var / tau2, 1e-3, 1e3)), "slope_sd": round(tau2**0.5, 4),
            "issuers_used": len(slopes), "estimated": True}
