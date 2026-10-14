import pytest

from robustness_gap.analyze import error_intervals
from robustness_gap.records import Status

TOL = 1e-6


def test_complete_resolved_case():
    e = error_intervals(L=0.10, U=0.30, a=0.199, b=0.201, tol=TOL)
    assert e.consistency_status is Status.SUCCESS
    assert (e.A, e.B) == (0.199, 0.201)
    assert e.total_gap == pytest.approx(0.20)
    assert e.certificate_error_lower == pytest.approx(0.099)
    assert e.certificate_error_upper == pytest.approx(0.101)
    assert e.attack_error_lower == pytest.approx(0.099)
    assert e.attack_error_upper == pytest.approx(0.101)
    assert e.certificate_error_lower + e.attack_error_upper == pytest.approx(e.total_gap)


def test_attack_tighter_than_reference_incumbent():
    e = error_intervals(L=0.1, U=0.2, a=0.15, b=0.25, tol=TOL)
    assert e.B == 0.2 and e.attack_error_lower == 0.0


def test_missing_U_never_enters_attack_error():
    e = error_intervals(L=0.1, U=None, a=0.2, b=0.21, tol=TOL)
    assert e.attack_error_lower is None and e.attack_error_upper is None and e.total_gap is None
    assert e.certificate_error_lower == pytest.approx(0.1)
    assert e.certificate_error_upper == pytest.approx(0.11)


def test_unresolved_reference_gives_wide_interval():
    e = error_intervals(L=0.1, U=0.3, a=0.12, b=None, tol=TOL)
    assert e.certificate_error_lower == pytest.approx(0.02)
    assert e.certificate_error_upper == pytest.approx(0.2)
    assert e.attack_error_lower == pytest.approx(0.0)


def test_inconsistent_bounds_are_exposed_not_clamped():
    e = error_intervals(L=0.3, U=0.2, a=None, b=None, tol=TOL)
    assert e.consistency_status is Status.INCONSISTENT_BOUNDS
    assert e.certificate_error_lower is None and e.attack_error_lower is None


def test_within_tolerance_is_consistent():
    e = error_intervals(L=0.2 + 1e-7, U=0.2, a=None, b=None, tol=TOL)
    assert e.consistency_status is Status.SUCCESS


def test_nothing_available():
    e = error_intervals(L=None, U=None, a=None, b=None, tol=TOL)
    assert e.consistency_status is Status.UNAVAILABLE
