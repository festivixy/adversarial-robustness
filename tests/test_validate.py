import numpy as np
import pytest

from robustness_gap.targets import TargetPolicy, TargetDefinition
from robustness_gap.validate import validate_candidate

from conftest import linear_net

NET = linear_net((1.0, 0.0), b1=-0.5)
X = np.array([0.4, 0.6])
STRICT = TargetDefinition(TargetPolicy.STRICT_PREDICTION_CHANGE)


def check(c, target=STRICT):
    return validate_candidate(NET, X, np.asarray(c, dtype=float), 0, target, domain_tol=1e-12)


def test_valid_candidate():
    v = check([0.6, 0.6])
    assert v.valid and v.recomputed_distance == pytest.approx(0.2) and v.predicted_class == 1


@pytest.mark.parametrize("cand,reason", [
    ([np.nan, 0.6], "non-finite"),
    ([np.inf, 0.6], "non-finite"),
    ([0.6, 0.6, 0.1], "bad shape"),
    ([1.2, 0.6], "outside domain"),
    ([0.45, 0.6], "target predicate"),
])
def test_invalid_candidates(cand, reason):
    v = check(cand)
    assert not v.valid and reason in v.reason


def test_tie_candidate_is_not_a_changed_label():
    v = check([0.5, 0.6])
    assert v.score_condition_satisfied and not v.prediction_changed and not v.valid
    assert check([0.5, 0.6], TargetDefinition(TargetPolicy.SCORE_BOUNDARY)).valid
