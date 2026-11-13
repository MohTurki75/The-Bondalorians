"""Univers synthétique pour tester toute la chaîne avant l'arrivée des données.

La structure des spreads est plausible (notation, maturité, séniorité, secteur,
effet émetteur, bruit) mais inventée : les erreurs mesurées dessus ne disent
rien de la précision sur les données réelles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bondalorians.data.schema import RATING_SCALE

SECTORS = {
    "Banques": 0.05, "Assurance": 0.10, "Services publics": -0.10, "Télécoms": 0.00,
    "Industrie": 0.02, "Énergie": 0.08, "Immobilier": 0.20, "Consommation": -0.03,
    "Santé": -0.08, "Technologie": -0.02,
}
COUNTRIES = ["FR", "DE", "IT", "ES", "NL", "US", "GB"]
FINANCIALS = {"Banques", "Assurance"}
SENIORITY_ADD = {  # (écart en log, crans de notation en plus)
    "SECURED": (-0.25, -1), "SENIOR_PREFERRED": (0.0, 0), "SENIOR_UNSECURED": (0.0, 0),
    "SENIOR_NON_PREFERRED": (0.22, 1), "TIER2": (0.55, 2), "SUBORDINATED": (0.50, 2),
    "AT1": (1.00, 4), "HYBRID": (0.80, 3),
}


def make_universe(n_bonds: int = 4000, n_issuers: int = 600, seed: int = 42,
                  noise_by_rating: float = 0.0) -> pd.DataFrame:
    """`noise_by_rating` > 0 rend le bruit croissant avec la dégradation de la
    notation (écart-type ajouté par cran au-delà de AA). Sert à vérifier que le
    modèle de difficulté détecte des cas inégalement prévisibles."""
    rng = np.random.default_rng(seed)
    sectors = list(SECTORS)
    issuers = pd.DataFrame({
        "issuer": [f"Émetteur {i:04d}" for i in range(n_issuers)],
        "sector": rng.choice(sectors, n_issuers),
        "country": rng.choice(COUNTRIES, n_issuers),
        "notch": np.clip(np.round(rng.normal(8.5, 3.0, n_issuers)), 1, 17).astype(int),
        "alpha": rng.normal(0.0, 0.15, n_issuers),
    })
    # Quelques gros émetteurs concentrent beaucoup d'obligations, d'autres une seule.
    popularity = np.minimum(rng.pareto(1.5, n_issuers) + 0.2, 6.0)
    pick = rng.choice(n_issuers, size=n_bonds, p=popularity / popularity.sum())

    rows = []
    for i, k in enumerate(pick):
        issuer = issuers.iloc[k]
        financial = issuer["sector"] in FINANCIALS
        if financial:
            seniority = rng.choice(
                ["SECURED", "SENIOR_PREFERRED", "SENIOR_NON_PREFERRED", "TIER2", "AT1"],
                p=[0.10, 0.45, 0.25, 0.14, 0.06],
            )
        else:
            seniority = rng.choice(
                ["SECURED", "SENIOR_UNSECURED", "SUBORDINATED", "HYBRID"], p=[0.06, 0.84, 0.05, 0.05]
            )
        add_log, add_notch = SENIORITY_ADD[seniority]
        notch = int(np.clip(issuer["notch"] + add_notch, 1, 19))
        maturity = float(np.clip(rng.gamma(2.2, 3.0), 0.5, 30.0))
        currency = rng.choice(["EUR", "USD", "GBP"], p=[0.75, 0.20, 0.05])
        callable_ = bool(seniority in ("AT1", "HYBRID", "TIER2") or rng.random() < 0.10)
        log_spread = (
            2.55 + 0.21 * notch + 0.0035 * notch**2
            + (0.42 - 0.012 * notch) * np.log(maturity)
            + add_log + SECTORS[issuer["sector"]]
            + {"EUR": 0.0, "USD": 0.06, "GBP": 0.10}[currency]
            + 0.05 * callable_
            + issuer["alpha"] + rng.normal(0.0, 0.08 + noise_by_rating * max(notch - 3, 0))
        )
        rows.append({
            "isin": f"XS{i:010d}",
            "issuer": issuer["issuer"], "sector": issuer["sector"], "country": issuer["country"],
            "currency": currency, "rating": RATING_SCALE[notch - 1], "seniority": seniority,
            "coupon_type": rng.choice(["FIXED", "FLOATING"], p=[0.9, 0.1]),
            "callable": callable_, "maturity_years": round(maturity, 2),
            "amount_outstanding": float(np.round(np.exp(rng.normal(20.0, 0.6)), -5)),
            "z_spread": round(float(np.exp(log_spread)), 1),
        })
    return pd.DataFrame(rows)
