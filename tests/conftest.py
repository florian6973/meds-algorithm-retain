"""Connect this algorithm to the shared acceptance suite."""

from pathlib import Path

import pytest

from meds_algorithm.conformance.cases import AlgorithmCase
from retain_meds.retain_meds import STUB_PROFILES, create_repository
from retain_meds.config import RETAINConfig

from .e2e import TEST_CONFIG, fitted_codes, prepare_meds_dev_run, run_e2e


@pytest.fixture(scope="session")
def algorithm_case():
    return AlgorithmCase(
        root=Path(__file__).resolve().parents[1],
        name="retain_meds",
        create_repository=create_repository,
        # Add representative overrides and supply any required constructor arguments.
        configurations=(RETAINConfig(), TEST_CONFIG),
        stub_profiles=STUB_PROFILES,
        run_e2e=run_e2e,
        fitted_codes=fitted_codes,
        prepare_meds_dev_run=prepare_meds_dev_run,
    )


@pytest.fixture(scope="session", params=create_repository().profiles)
def profile(request):
    return request.param
