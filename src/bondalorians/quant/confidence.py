"""Confiance apprise : un modèle prédit la difficulté d'un cas.

Au lieu de pondérer à la main des indicateurs, on apprend sur le backtest
l'erreur absolue (en log) du modèle retenu en fonction de la qualité des
comparables, du poids de l'émetteur, du désaccord entre modèles, etc.

Cette difficulté prédite sert deux fois :
- elle module la largeur de l'intervalle conforme (cas facile : intervalle
  étroit ; cas difficile : intervalle large) ;
- elle donne le score : part des cas du backtest jugés plus difficiles.
"""

from __future__ import annotations

import numpy as np

ERROR_FEATURES = [
    "n_eff", "mean_distance", "lam", "log_n_issuer", "filter_level", "notch", "log_t",
    "sen_rank", "log_segment", "gap_ab", "corr_abs", "unseen",
    "peer_disp", "issuer_disp", "issuer_disp_known",
]
_EPS = 1e-3


def error_features(t: dict, core: dict, log_b: float | None, log_c: float | None,
                   dispersion: dict | None = None) -> dict:
    """Indicateurs de difficulté d'un cas.

    `dispersion` (si fourni) active les indicateurs de dispersion : désaccord des
    comparables entre eux et dispersion des obligations de l'émetteur autour de sa
    courbe. Sinon ils valent zéro et n'apportent aucune information.
    """
    if dispersion is None:
        extra = {"peer_disp": 0.0, "issuer_disp": 0.0, "issuer_disp_known": 0.0}
    else:
        issuer_disp = core.get("issuer_disp", float("nan"))
        known = issuer_disp == issuer_disp  # faux pour NaN
        peer_disp = core.get("peer_disp", float("nan"))
        extra = {
            "peer_disp": peer_disp if peer_disp == peer_disp else dispersion["noise_sd"],
            "issuer_disp": issuer_disp if known else dispersion["noise_sd"],
            "issuer_disp_known": 1.0 if known else 0.0,
        }
    return {**extra, **{
        "n_eff": core["n_eff"], "mean_distance": core["mean_distance"], "lam": core["lam"],
        "log_n_issuer": float(np.log1p(core["n_issuer"])), "filter_level": float(core["filter_level"]),
        "notch": t["notch"], "log_t": t["log_t"], "sen_rank": t["sen_rank"],
        "log_segment": float(np.log1p(core["segment_count"])),
        "gap_ab": 0.0 if log_b is None else abs(core["log_a"] - log_b),
        "corr_abs": 0.0 if log_c is None else abs(log_c - core["log_a"]),
        "unseen": 0.0 if core["issuer_seen"] else 1.0,
    }}


class ErrorModel:
    """Prédit log(|erreur| + eps). CatBoost si disponible, sinon régression ridge."""

    def __init__(self, use_catboost: bool, seed: int = 7):
        self.use_catboost = use_catboost
        self.seed = seed

    def fit(self, x: np.ndarray, abs_error: np.ndarray) -> "ErrorModel":
        y = np.log(np.asarray(abs_error, dtype=float) + _EPS)
        if self.use_catboost:
            from catboost import CatBoostRegressor

            self.model_ = CatBoostRegressor(
                iterations=150, depth=3, learning_rate=0.05, loss_function="RMSE",
                random_seed=self.seed, verbose=False, allow_writing_files=False,
            )
            self.model_.fit(x, y)
        else:
            self.mean_, self.std_ = x.mean(0), x.std(0) + 1e-9
            z = np.column_stack([np.ones(len(x)), (x - self.mean_) / self.std_])
            penalty = np.eye(z.shape[1])
            penalty[0, 0] = 0.0
            self.coef_ = np.linalg.solve(z.T @ z + penalty, z.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Erreur typique attendue, en log du spread (donc à peu près en relatif)."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if self.use_catboost:
            log_error = np.asarray(self.model_.predict(x), dtype=float)
        else:
            z = np.column_stack([np.ones(len(x)), (x - self.mean_) / self.std_])
            log_error = z @ self.coef_
        return np.maximum(np.exp(log_error), _EPS)


def confidence_from_difficulty(sigma: float, sigma_reference: np.ndarray, estimate_bp: float,
                               drivers: dict, grades: dict) -> dict:
    """Score = pourcentage des cas du backtest jugés plus difficiles que celui-ci."""
    rank = np.searchsorted(sigma_reference, sigma, side="right") / len(sigma_reference)
    score = round(100 * (1 - rank), 1)
    grade = "A" if score >= grades["A"] else "B" if score >= grades["B"] else "C"
    return {
        "score": score, "grade": grade,
        "typical_error_pct": round(100 * sigma, 1),
        "typical_error_bp": round(estimate_bp * sigma, 1),
        "drivers": drivers,
    }
