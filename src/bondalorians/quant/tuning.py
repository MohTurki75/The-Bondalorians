"""Calibrage des comparables : poids de distance, largeur du noyau, nombre retenu.

On cherche les paramètres qui minimisent l'erreur hors échantillon du modèle A.
Garde-fous contre le surapprentissage du backtest :
- les paramètres sont réglés sur un sous-ensemble d'obligations ;
- ils ne sont adoptés que s'ils font mieux que les valeurs par défaut sur un
  second sous-ensemble, jamais utilisé pour le réglage ;
- les obligations de réglage sont ensuite exclues du calcul des intervalles ;
- chaque critère garde un poids minimal, pour que la sélection reste défendable
  devant un analyste (une notation ignorée ne se justifie pas à l'oral).
"""

from __future__ import annotations

import numpy as np

WEIGHT_KEYS = ("notch", "maturity", "sector", "country", "size")
TOP_K_GRID = (5, 10, 15, 25, 40, 60)


def _case_error(case: dict, weights: np.ndarray, bandwidth: float, top_k: int) -> float:
    if case["components"].shape[0] == 0:
        return abs(case["base"] - case["y"])
    d = case["components"] @ weights
    if len(d) > top_k:
        pick = np.argpartition(d, top_k)[:top_k]
        d, resid, obs = d[pick], case["resid"][pick], case["obs"][pick]
    else:
        resid, obs = case["resid"], case["obs"]
    w = np.exp(-((d / bandwidth) ** 2)) * obs
    total = w.sum()
    r_peers = float((w * resid).sum() / total) if total > 0 else 0.0
    return abs(case["base"] + case["one_minus_lam"] * r_peers - case["y"])


def _objective(cases: list[dict], params: dict) -> float:
    weights = np.array([params["weights"][k] for k in WEIGHT_KEYS], dtype=np.float32)
    errors = [_case_error(c, weights, params["bandwidth"], params["top_k"]) for c in cases]
    return float(np.mean(errors))


def _floored(weights: np.ndarray, floor: float) -> np.ndarray:
    """Impose un poids minimal à chaque critère : aucun n'est ignoré."""
    weights = weights / weights.sum()
    return floor + (1 - floor * len(weights)) * weights


def _random_params(rng, floor: float) -> dict:
    weights = _floored(rng.dirichlet(np.full(len(WEIGHT_KEYS), 1.5)), floor)
    return {
        "weights": dict(zip(WEIGHT_KEYS, weights.tolist())),
        "bandwidth": float(np.exp(rng.uniform(np.log(0.05), np.log(0.8)))),
        "top_k": int(rng.choice(TOP_K_GRID)),
    }


def _perturb(params: dict, rng, floor: float) -> dict:
    weights = np.array([params["weights"][k] for k in WEIGHT_KEYS]) * np.exp(rng.normal(0, 0.25, len(WEIGHT_KEYS)))
    weights = _floored(weights, floor)
    position = TOP_K_GRID.index(params["top_k"]) + int(rng.integers(-1, 2))
    return {
        "weights": dict(zip(WEIGHT_KEYS, weights.tolist())),
        "bandwidth": float(np.clip(params["bandwidth"] * np.exp(rng.normal(0, 0.2)), 0.03, 1.0)),
        "top_k": TOP_K_GRID[int(np.clip(position, 0, len(TOP_K_GRID) - 1))],
    }


def tune_comparables(estimator, trials: int, share: float, seed: int, floor: float = 0.05) -> dict:
    """Règle les paramètres des comparables de `estimator` (déjà ajusté, sans ML).

    Renvoie un rapport ; met à jour `estimator.params_` seulement si le gain est
    confirmé sur l'échantillon de validation.
    """
    df = estimator.df_
    rng = np.random.default_rng(seed)
    n = len(df)
    order = rng.permutation(n)
    size = max(1, int(share * n))
    tune_ids, valid_ids = set(order[:size].tolist()), set(order[size:2 * size].tolist())
    rows = df.to_dict("records")

    cases = {"tune": [], "valid": []}
    for folds in estimator.make_folds(rng).values():
        for fold in np.unique(folds):
            test = [i for i in np.flatnonzero(folds == fold) if i in tune_ids or i in valid_ids]
            if not test:
                continue
            sub = estimator.spawn(use_ml=False)
            sub.fit_base(df[folds != fold])
            peer_logs = sub.peer_.predict_log(df.iloc[test])
            for i, peer_log in zip(test, peer_logs):
                cand = sub.candidates(rows[i], peer_log=float(peer_log))
                idx = cand["idx"]
                cases["tune" if i in tune_ids else "valid"].append({
                    "y": float(np.log(rows[i]["z_spread"])),
                    "base": cand["peer_log"] + cand["lam"] * cand["r_issuer"],
                    "one_minus_lam": 1 - cand["lam"],
                    "components": cand["components"].astype(np.float32),
                    "resid": sub.a_["resid"][idx], "obs": sub.a_["obs_weight"][idx],
                })

    default = {k: estimator.params_[k] for k in ("weights", "bandwidth", "top_k")}
    if default["top_k"] not in TOP_K_GRID:
        default = {**default, "top_k": min(TOP_K_GRID, key=lambda v: abs(v - default["top_k"]))}
    best, best_error = default, _objective(cases["tune"], default)
    default_tune_error = best_error
    n_random = int(0.7 * trials)
    for trial in range(trials):
        candidate = _random_params(rng, floor) if trial < n_random else _perturb(best, rng, floor)
        error = _objective(cases["tune"], candidate)
        if error < best_error:
            best, best_error = candidate, error

    default_valid, tuned_valid = _objective(cases["valid"], default), _objective(cases["valid"], best)
    adopted = tuned_valid < default_valid
    if adopted:
        estimator.params_ = best
    return {
        "adopted": adopted, "trials": trials,
        "n_tune_cases": len(cases["tune"]), "n_valid_cases": len(cases["valid"]),
        # Erreur absolue moyenne en log, exprimée en pourcentage.
        "tune_error_pct": {"default": round(100 * default_tune_error, 2), "tuned": round(100 * best_error, 2)},
        "valid_error_pct": {"default": round(100 * default_valid, 2), "tuned": round(100 * tuned_valid, 2)},
        "params": {
            "weights": {k: round(v, 3) for k, v in best["weights"].items()},
            "bandwidth": round(best["bandwidth"], 3), "top_k": best["top_k"],
        },
        "tune_row_ids": sorted(tune_ids),
    }
