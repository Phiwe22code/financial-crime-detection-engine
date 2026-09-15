"""Tests for continuous risk-band boundaries."""

from src.scoring import _assign_band


def test_decimal_scores_do_not_fall_through_band_gaps():
    assert _assign_band(39.9) == "Low"
    assert _assign_band(40.0) == "Medium"
    assert _assign_band(69.9) == "Medium"
    assert _assign_band(70.0) == "High"
