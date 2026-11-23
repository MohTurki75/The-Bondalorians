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


PROFILES = ("simple", "realiste")
# Profil « réaliste » : effets que la courbe des pairs d'origine ne sait pas reproduire.
SECTOR_RATING_SLOPE = {  # supplément de pente de notation par secteur (en log, par cran au-delà de BBB)
    "Immobilier": 0.04, "Énergie": 0.03, "Banques": 0.02, "Assurance": 0.01,
    "Services publics": -0.02, "Santé": -0.01,
}
COUNTRY_ADD = {"IT": 0.08, "ES": 0.04, "GB": 0.02}


def make_universe(n_bonds: int = 4000, n_issuers: int = 600, seed: int = 42,
                  noise_by_rating: float = 0.0, profile: str = "simple") -> pd.DataFrame:
    """Univers synthétique.

    - `profile="simple"` : modèle log-linéaire très proche de la courbe des pairs ;
      il favorise le modèle A par construction.
    - `profile="realiste"` : ajoute ce qu'un marché réel montre et que le modèle
      d'origine ne capture pas : courbes qui s'aplatissent en haut rendement,
      sensibilité à la notation propre au secteur, prime de subordination plus
      forte pour les émetteurs faibles, pente de maturité propre à chaque émetteur,
      effet pays, prime d'illiquidité des petites souches, bruit plus fort pour
      les notations faibles et la dette subordonnée, et 4 % de notations « en
      retard » sur le marché.
    - `noise_by_rating` > 0 (profil simple) : bruit croissant avec la notation.
    """
    if profile not in PROFILES:
        raise ValueError(f"Profil inconnu : {profile} (attendu : {PROFILES}).")
    realistic = profile == "realiste"
    rng = np.random.default_rng(seed)
    sectors = list(SECTORS)
    issuers = pd.DataFrame({
        "issuer": [f"Émetteur {i:04d}" for i in range(n_issuers)],
        "sector": rng.choice(sectors, n_issuers),
        "country": rng.choice(COUNTRIES, n_issuers),
        "notch": np.clip(np.round(rng.normal(8.5, 3.0, n_issuers)), 1, 17).astype(int),
        "alpha": rng.normal(0.0, 0.15, n_issuers),
        "beta": rng.normal(0.0, 0.08, n_issuers),  # pente de maturité propre (profil réaliste)
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
        log_amt = rng.normal(20.0, 0.6)
        log_t = np.log(maturity)
        if not realistic:
            log_spread = (
                2.55 + 0.21 * notch + 0.0035 * notch**2
                + (0.42 - 0.012 * notch) * log_t
                + add_log + SECTORS[issuer["sector"]]
                + {"EUR": 0.0, "USD": 0.06, "GBP": 0.10}[currency]
                + 0.05 * callable_
                + issuer["alpha"] + rng.normal(0.0, 0.08 + noise_by_rating * max(notch - 3, 0))
            )
        else:
            subordinated = seniority in ("TIER2", "AT1", "SUBORDINATED", "HYBRID")
            # Courbe qui s'aplatit (et se bombe) quand la notation se dégrade.
            term = (0.42 - 0.03 * max(notch - 10, 0)) * log_t - 0.05 * log_t**2
            # La prime de subordination grossit quand l'émetteur est faible.
            seniority_add = add_log * max(0.3, 1 + 0.08 * (issuer["notch"] - 9))
            noise_sd = 0.05 + 0.012 * max(notch - 6, 0) + 0.08 * subordinated + 0.03 * callable_
            stale = rng.random()
            log_spread = (
                2.55 + 0.21 * notch + 0.0035 * notch**2 + term + seniority_add
                + SECTORS[issuer["sector"]] + SECTOR_RATING_SLOPE.get(issuer["sector"], 0.0) * (notch - 9)
                + {"EUR": 0.0, "USD": 0.06, "GBP": 0.10}[currency] + COUNTRY_ADD.get(issuer["country"], 0.0)
                + 0.05 * callable_ - 0.06 * (log_amt - 20.0)
                + issuer["alpha"] + issuer["beta"] * (log_t - np.log(5.0))
                + (0.35 if stale < 0.03 else -0.25 if stale < 0.04 else 0.0)
                + rng.normal(0.0, noise_sd)
            )
        rows.append({
            "isin": f"XS{i:010d}",
            "issuer": issuer["issuer"], "sector": issuer["sector"], "country": issuer["country"],
            "currency": currency, "rating": RATING_SCALE[notch - 1], "seniority": seniority,
            "coupon_type": rng.choice(["FIXED", "FLOATING"], p=[0.9, 0.1]),
            "callable": callable_, "maturity_years": round(maturity, 2),
            "amount_outstanding": float(np.round(np.exp(log_amt), -5)),
            "z_spread": round(float(np.exp(log_spread)), 1),
        })
    return pd.DataFrame(rows)


MOODYS = {"AAA": "Aaa", "AA+": "Aa1", "AA": "Aa2", "AA-": "Aa3", "A+": "A1", "A": "A2", "A-": "A3",
          "BBB+": "Baa1", "BBB": "Baa2", "BBB-": "Baa3", "BB+": "Ba1", "BB": "Ba2", "BB-": "Ba3",
          "B+": "B1", "B": "B2", "B-": "B3", "CCC+": "Caa1", "CCC": "Caa2", "CCC-": "Caa3"}
RAW_SENIORITY = {"SECURED": "Secured", "SENIOR_PREFERRED": "Sr Preferred", "SENIOR_UNSECURED": "Sr Unsecured",
                 "SENIOR_NON_PREFERRED": "Senior Non-Preferred", "TIER2": "Tier 2", "AT1": "AT1",
                 "SUBORDINATED": "Jr Subordinated", "HYBRID": "Hybrid perpetual"}


def make_raw_database(n_bonds: int = 3000, seed: int = 11, as_of: str = "2026-11-24",
                      esg_effect: float = 0.04) -> pd.DataFrame:
    """Base « brute » comme celle d'un jour 1 : noms de colonnes libres, formats hétérogènes.

    Contient en plus une note ESG (lettres MSCI) qui réduit le spread de `esg_effect`
    (en log) par cran, une date d'émission, une duration, un prix et un rendement
    calculés à partir du spread (donc liés mécaniquement à lui).
    """
    rng = np.random.default_rng(seed)
    base = make_universe(n_bonds=n_bonds, n_issuers=450, seed=seed, profile="realiste")
    esg_letters = np.array(["CCC", "B", "BB", "BBB", "A", "AA", "AAA"])
    issuer_esg = {i: rng.integers(0, 7) for i in base["issuer"].unique()}
    esg = base["issuer"].map(issuer_esg).to_numpy()
    spread = base["z_spread"].to_numpy() * np.exp(-esg_effect * (esg - 3))
    today = pd.Timestamp(as_of)
    maturity = today + pd.to_timedelta(base["maturity_years"].to_numpy() * 365.25, unit="D")
    age = rng.uniform(0.1, 8.0, len(base))
    issue = today - pd.to_timedelta(age * 365.25, unit="D")
    swap = {"EUR": 2.4, "USD": 4.1, "GBP": 4.0}
    ytm = base["currency"].map(swap).to_numpy() + spread / 100
    coupon = np.round(ytm + rng.normal(0, 0.6, len(base)), 3)
    t = base["maturity_years"].to_numpy()
    duration = (1 - (1 + ytm / 100) ** -t) / (ytm / 100)
    price = coupon / (ytm / 100) * (1 - (1 + ytm / 100) ** -t) + 100 * (1 + ytm / 100) ** -t
    ratings = [MOODYS.get(r, r) if rng.random() < 0.3 else r for r in base["rating"]]
    return pd.DataFrame({
        "Code ISIN": base["isin"], "Nom émetteur": base["issuer"], "Secteur GICS": base["sector"],
        "Pays de risque": base["country"], "Devise": base["currency"], "Notation composite": ratings,
        "Rang de paiement": base["seniority"].map(RAW_SENIORITY), "Type coupon": np.where(
            base["coupon_type"] == "FIXED", "Fixe", "Variable"),
        "Coupon (%)": coupon, "Callable": np.where(base["callable"], "Y", "N"),
        "Date échéance": maturity.strftime("%d/%m/%Y"), "Date émission": issue.strftime("%d/%m/%Y"),
        "Encours": base["amount_outstanding"], "Note ESG MSCI": esg_letters[esg],
        "Duration modifiée": np.round(duration, 3), "Prix": np.round(price, 3),
        "Rendement (%)": np.round(ytm, 3), "Z-Spread (pb)": np.round(spread, 1),
    })
