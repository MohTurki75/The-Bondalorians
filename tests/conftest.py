import pytest

from bondalorians.quant.estimator import ZSpreadEstimator
from bondalorians.synthetic import make_universe


@pytest.fixture(scope="session")
def universe():
    return make_universe(n_bonds=1500, n_issuers=220, seed=3)


@pytest.fixture(scope="session")
def estimator(universe):
    return ZSpreadEstimator(use_ml=False).fit(universe)


@pytest.fixture()
def new_bond():
    return {
        "issuer": "Émetteur jamais vu", "sector": "Industrie", "country": "FR",
        "currency": "EUR", "rating": "BBB", "seniority": "SENIOR_UNSECURED",
        "maturity_years": 6.0,
    }
