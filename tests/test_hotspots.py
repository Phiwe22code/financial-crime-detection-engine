"""Tests for South African hotspot reference data."""

import pytest

from src.hotspots import city_locations, province_hotspots, station_hotspots


def test_all_nine_provinces_are_present():
    provinces = province_hotspots()
    assert len(provinces) == 9
    assert provinces["province"].nunique() == 9
    assert provinces["credit_share"].sum() == pytest.approx(100.0)
    assert provinces["debit_share"].sum() == pytest.approx(100.0)


def test_every_province_has_a_simulation_city():
    provinces = set(province_hotspots()["province"])
    cities = city_locations()
    assert provinces == set(cities["province"])
    assert cities[["lat", "lon"]].notna().all().all()


def test_station_hotspots_have_source_metadata():
    stations = station_hotspots()
    assert not stations.empty
    assert (stations["cases"] > 0).all()
    assert stations[["source", "period"]].notna().all().all()
