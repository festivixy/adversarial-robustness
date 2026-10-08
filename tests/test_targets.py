import numpy as np
import pytest

from robustness_gap.targets import (TargetPolicy, TargetDefinition, UnresolvedTargetError,
                                    evaluate_target, predict)

SCORE = TargetDefinition(TargetPolicy.SCORE_BOUNDARY)
STRICT = TargetDefinition(TargetPolicy.STRICT_PREDICTION_CHANGE)
MARGIN = TargetDefinition(TargetPolicy.POSITIVE_MARGIN, margin=0.01)


def test_tie_predicts_lowest_index():
    assert predict(np.array([1.0, 1.0])) == 0
    assert predict(np.array([1.0, 2.0])) == 1


def test_unresolved_target_refuses():
    with pytest.raises(UnresolvedTargetError):
        evaluate_target(TargetDefinition(None), np.array([0.0, 1.0]), 0)


def test_margin_validation():
    with pytest.raises(ValueError):
        TargetDefinition(TargetPolicy.POSITIVE_MARGIN, margin=0.0)
    with pytest.raises(ValueError):
        TargetDefinition(TargetPolicy.SCORE_BOUNDARY, margin=0.1)


def test_isolated_tie_original_class_0():
    tie = np.array([0.3, 0.3])
    s = evaluate_target(SCORE, tie, original_class=0)
    assert s.score_condition_satisfied and not s.prediction_changed and s.target_satisfied
    t = evaluate_target(STRICT, tie, original_class=0)
    assert t.score_condition_satisfied and not t.prediction_changed and not t.target_satisfied
    assert not evaluate_target(MARGIN, tie, original_class=0).target_satisfied


def test_isolated_tie_original_class_1():
    tie = np.array([0.3, 0.3])
    assert evaluate_target(SCORE, tie, 1).prediction_changed
    assert evaluate_target(STRICT, tie, 1).target_satisfied


def test_flat_tie_region_constant_classifier():
    for x in np.random.default_rng(0).random((50, 2)):
        logits = np.array([2.0, 2.0])
        assert evaluate_target(SCORE, logits, 0).target_satisfied
        assert not evaluate_target(STRICT, logits, 0).target_satisfied


def test_target_ids_distinct():
    ids = {SCORE.target_id, STRICT.target_id, MARGIN.target_id, TargetDefinition(None).target_id}
    assert len(ids) == 4
