"""The counting-baseline scorer, whose numbers are meant to be comparable across ports.

`average_precision` here must equal `sklearn.metrics.average_precision_score`, because the analysis
session computes the cross-port table with sklearn and TECO's baseline reports the same quantity. The
reference values below were taken from sklearn and are hardcoded rather than compared live, so the
agreement survives sklearn not being a declared dependency of this port.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "counting_baseline", Path(__file__).parents[1] / "scripts" / "counting-baseline.py"
)
counting_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(counting_baseline)


@pytest.mark.parametrize(
    "labels,scores,expected",
    [
        ([1, 0, 1], [0.9, 0.8, 0.7], 0.8333333333),
        ([1, 1, 0, 0], [0.5, 0.5, 0.5, 0.5], 0.5000000000),  # all tied: collapses to one threshold
        ([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], 1.0000000000),  # perfectly separated
        ([1, 0, 0, 0], [0.9, 0.1, 0.1, 0.1], 1.0000000000),  # single positive ranked first
    ],
)
def test_average_precision_matches_sklearn(labels, scores, expected):
    got = counting_baseline.average_precision(np.array(labels), np.array(scores))
    assert got == pytest.approx(expected, abs=1e-9)


def test_average_precision_is_the_prevalence_for_a_constant_score():
    """A constant predictor scores its own base rate — the floor the AUPRC lift is measured against."""
    labels = np.array([1] * 3 + [0] * 7)
    constant = np.full(labels.size, 0.42)
    assert counting_baseline.average_precision(labels, constant) == pytest.approx(0.3, abs=1e-9)


def test_ties_are_collapsed_rather_than_ordered():
    """Tied scores must not be silently broken by array order, which would inflate the metric."""
    labels = np.array([1, 0, 1, 0])
    tied = np.full(4, 0.5)
    assert counting_baseline.average_precision(labels, tied) == pytest.approx(0.5, abs=1e-9)
    reordered = counting_baseline.average_precision(labels[::-1], tied)
    assert reordered == pytest.approx(0.5, abs=1e-9)
