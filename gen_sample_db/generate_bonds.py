#!/usr/bin/env python3
"""Generate a reproducible synthetic corporate-bond universe as CSV.

The generated Z-spreads are illustrative synthetic values, not market data.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


COLUMNS = [
    "isin",
    "issuer",
    "sector",
    "country",
    "currency",
    "rating",
    "seniority",
    "coupon_type",
    "callable",
    "maturity_years",
    "amount_outstanding",
    "z_spread",
]

SECTORS = [
    "Énergie", "Santé", "Services publics", "Télécommunications",
    "Technologie", "Industrie", "Finance", "Consommation", "Transport",
]
COUNTRIES = ["FR", "DE", "NL", "BE", "ES", "IT", "US", "GB", "CH"]
CURRENCIES_BY_COUNTRY = {
    "FR": ["EUR"], "DE": ["EUR"], "NL": ["EUR"], "BE": ["EUR"],
    "ES": ["EUR"], "IT": ["EUR"], "US": ["USD"], "GB": ["GBP"],
    "CH": ["CHF"],
}
RATINGS = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B"]
# Approximate rating notch: lower credit quality generally means higher spread.
RATING_NOTCH = {rating: i for i, rating in enumerate(RATINGS)}
SENIORITIES = ["SENIOR_SECURED", "SENIOR_UNSECURED", "SUBORDINATED", "JUNIOR_SUBORDINATED"]
COUPON_TYPES = ["FIXED", "FLOATING", "INFLATION_LINKED"]

# Illustrative country/currency baselines in basis points.
CURRENCY_BASE_BPS = {"EUR": 18.0, "USD": 22.0, "GBP": 24.0, "CHF": 12.0}
SENIORITY_ADD_BPS = {
    "SENIOR_SECURED": -8.0,
    "SENIOR_UNSECURED": 0.0,
    "SUBORDINATED": 55.0,
    "JUNIOR_SUBORDINATED": 100.0,
}
COUPON_ADD_BPS = {"FIXED": 0.0, "FLOATING": 8.0, "INFLATION_LINKED": 12.0}


def triangular_int(rng: random.Random, low: int, mode: int, high: int) -> int:
    return int(round(rng.triangular(low, high, mode)))


def maturity_effect(years: float) -> float:
    """A mild, non-linear term effect, in basis points."""
    return 3.0 * years + 1.1 * max(years - 7.0, 0.0) ** 1.35


def generate_row(index: int, rng: random.Random, regime_bps: float) -> dict[str, object]:
    # Group bonds into issuers so that multiple securities share issuer risk.
    issuer_number = rng.randint(1, 420)
    issuer = f"Émetteur {issuer_number:04d}"
    issuer_spread_effect = (issuer_number % 17 - 8) * 2.4 + rng.gauss(0, 5.0)

    country = rng.choice(COUNTRIES)
    currency = rng.choice(CURRENCIES_BY_COUNTRY[country])
    sector = rng.choice(SECTORS)
    rating = rng.choices(RATINGS, weights=[1, 1, 2, 2, 3, 4, 5, 6, 8, 7, 5, 5, 4, 3, 2])[0]
    seniority = rng.choices(SENIORITIES, weights=[2, 7, 2, 1])[0]
    coupon_type = rng.choices(COUPON_TYPES, weights=[8, 2, 1])[0]
    callable_bond = rng.random() < 0.18
    maturity = round(rng.triangular(0.5, 30.0, 6.0), 2)
    amount = float(round(10 ** rng.uniform(7.0, 9.35), -5))

    rating_effect = RATING_NOTCH[rating] * 21.0
    sector_effect = {
        "Services publics": -8.0,
        "Santé": -3.0,
        "Finance": 10.0,
        "Énergie": 12.0,
        "Transport": 8.0,
        "Technologie": 3.0,
    }.get(sector, 0.0)
    callable_effect = 18.0 if callable_bond else 0.0
    size_effect = max(0.0, 8.0 - (amount / 1_000_000_000.0) * 5.0)

    # Add modest noise and rare outliers for robust-comparator testing.
    noise = rng.gauss(0.0, 13.0)
    if rng.random() < 0.015:
        noise += rng.choice([-1, 1]) * rng.uniform(35, 90)

    spread = (
        CURRENCY_BASE_BPS[currency]
        + rating_effect
        + maturity_effect(maturity)
        + SENIORITY_ADD_BPS[seniority]
        + COUPON_ADD_BPS[coupon_type]
        + callable_effect
        + sector_effect
        + issuer_spread_effect
        + size_effect
        + regime_bps
        + noise
    )
    spread = max(5.0, min(spread, 1_500.0))

    return {
        "isin": f"XS{index:010d}",
        "issuer": issuer,
        "sector": sector,
        "country": country,
        "currency": currency,
        "rating": rating,
        "seniority": seniority,
        "coupon_type": coupon_type,
        "callable": callable_bond,
        "maturity_years": maturity,
        "amount_outstanding": amount,
        "z_spread": round(spread, 1),
    }


def generate_dataset(count: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    # Shared market regime creates co-movement across all rows in this snapshot.
    regime_bps = rng.uniform(-20.0, 35.0)
    return [generate_row(i, rng, regime_bps) for i in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic bond data as CSV.")
    parser.add_argument("--rows", type=int, default=4000, help="Number of bonds (default: 4000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=Path, default=Path("bonds_synthetic.csv"), help="Output CSV path")
    args = parser.parse_args()

    if args.rows < 1:
        parser.error("--rows must be at least 1")

    rows = generate_dataset(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} synthetic bonds: {args.output}")
    print("Reminder: z_spread values are synthetic and for testing only.")


if __name__ == "__main__":
    main()
