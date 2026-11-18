import numpy as np
import pytest

from contactwise.metrics import precision_at_k, scenario_value, summarize, top_k_indices


def test_top_k_and_ties_are_deterministic():
    y = np.array([0, 1, 1, 0])
    p = [0.3, 0.9, 0.8, 0.8]
    assert list(top_k_indices(p, 2)) == [1, 2]
    assert precision_at_k(y, p, 2) == 1.0
    assert precision_at_k(y, p, 20) == 0.5


def test_summarize_and_noncausal_scenario():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.8, 0.9, 0.2])
    result = summarize(y, p, 2)
    assert result["precision_at_k"] == 1.0
    assert result["lift_at_k"] == 2.0
    assert result["observed_positives_at_k"] == 2
    scenario = scenario_value(y, p, 2, 100, 5)
    assert scenario["historical_outcome_scenario_proxy"] == 190.0


def test_bad_budget_rejected():
    with pytest.raises(ValueError):
        top_k_indices([0.1], 0)
