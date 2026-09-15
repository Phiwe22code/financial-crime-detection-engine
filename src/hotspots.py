"""South African fraud hotspot reference data and map helpers.

Provincial card-fraud shares are transcribed from SABRIC's 2024 Annual Crime
Statistics report.  Station values are selected commercial-crime counts from
SAPS' January--March 2025 release.  They are historical context, not a live
crime feed and not an input to individual customer risk decisions.
"""

from __future__ import annotations

import pandas as pd


SABRIC_SOURCE = (
    "https://www.sabric.co.za/wp-content/uploads/2025/09/"
    "CRIME-STATISTICS-REPORT-2024.pdf"
)
SAPS_SOURCE = (
    "https://www.saps.gov.za/services/downloads/2024/"
    "2024-2025_Q4_crime_stats.pdf"
)


PROVINCE_HOTSPOTS = (
    {"province": "Gauteng", "code": "ZA-GP", "lat": -26.2708, "lon": 28.1123, "credit_share": 50.2, "debit_share": 41.9},
    {"province": "Western Cape", "code": "ZA-WC", "lat": -33.2278, "lon": 21.8569, "credit_share": 33.6, "debit_share": 18.0},
    {"province": "KwaZulu-Natal", "code": "ZA-KZN", "lat": -28.5306, "lon": 30.8958, "credit_share": 10.7, "debit_share": 12.7},
    {"province": "Limpopo", "code": "ZA-LP", "lat": -23.4013, "lon": 29.4179, "credit_share": 1.6, "debit_share": 7.9},
    {"province": "Mpumalanga", "code": "ZA-MP", "lat": -25.5653, "lon": 30.5279, "credit_share": 1.6, "debit_share": 7.6},
    {"province": "North West", "code": "ZA-NW", "lat": -26.6639, "lon": 25.2838, "credit_share": 0.5, "debit_share": 3.8},
    {"province": "Free State", "code": "ZA-FS", "lat": -28.4541, "lon": 26.7968, "credit_share": 0.8, "debit_share": 3.2},
    {"province": "Eastern Cape", "code": "ZA-EC", "lat": -32.2968, "lon": 26.4194, "credit_share": 0.9, "debit_share": 4.4},
    {"province": "Northern Cape", "code": "ZA-NC", "lat": -29.0467, "lon": 21.8569, "credit_share": 0.1, "debit_share": 0.5},
)


CITY_LOCATIONS = (
    {"city": "Johannesburg", "province": "Gauteng", "lat": -26.2041, "lon": 28.0473},
    {"city": "Pretoria", "province": "Gauteng", "lat": -25.7479, "lon": 28.2293},
    {"city": "Midrand", "province": "Gauteng", "lat": -25.9992, "lon": 28.1263},
    {"city": "Sandton", "province": "Gauteng", "lat": -26.1076, "lon": 28.0567},
    {"city": "Cape Town", "province": "Western Cape", "lat": -33.9249, "lon": 18.4241},
    {"city": "Stellenbosch", "province": "Western Cape", "lat": -33.9321, "lon": 18.8602},
    {"city": "Durban", "province": "KwaZulu-Natal", "lat": -29.8587, "lon": 31.0218},
    {"city": "Pietermaritzburg", "province": "KwaZulu-Natal", "lat": -29.6006, "lon": 30.3794},
    {"city": "Polokwane", "province": "Limpopo", "lat": -23.9045, "lon": 29.4689},
    {"city": "Mbombela", "province": "Mpumalanga", "lat": -25.4753, "lon": 30.9694},
    {"city": "Rustenburg", "province": "North West", "lat": -25.6676, "lon": 27.2421},
    {"city": "Bloemfontein", "province": "Free State", "lat": -29.0852, "lon": 26.1596},
    {"city": "Gqeberha", "province": "Eastern Cape", "lat": -33.9608, "lon": 25.6022},
    {"city": "East London", "province": "Eastern Cape", "lat": -33.0153, "lon": 27.9116},
    {"city": "Kimberley", "province": "Northern Cape", "lat": -28.7282, "lon": 24.7499},
)


SAPS_STATION_HOTSPOTS = (
    {"station": "Midrand", "district": "Johannesburg", "province": "Gauteng", "lat": -25.9992, "lon": 28.1263, "cases": 564},
    {"station": "Sandton", "district": "Johannesburg", "province": "Gauteng", "lat": -26.1076, "lon": 28.0567, "cases": 471},
    {"station": "Cape Town Central", "district": "Cape Town", "province": "Western Cape", "lat": -33.9258, "lon": 18.4232, "cases": 427},
    {"station": "Lyttelton", "district": "Tshwane", "province": "Gauteng", "lat": -25.8277, "lon": 28.2012, "cases": 411},
    {"station": "Durban Central", "district": "eThekwini", "province": "KwaZulu-Natal", "lat": -29.8587, "lon": 31.0218, "cases": 380},
    {"station": "Brooklyn", "district": "Tshwane", "province": "Gauteng", "lat": -25.7714, "lon": 28.2336, "cases": 313},
)


def province_hotspots() -> pd.DataFrame:
    """Return SABRIC provincial card-fraud shares."""
    frame = pd.DataFrame(PROVINCE_HOTSPOTS)
    frame["source"] = "SABRIC Annual Crime Statistics 2024"
    frame["period"] = "2024"
    return frame


def city_locations() -> pd.DataFrame:
    """Return supported South African simulation locations."""
    return pd.DataFrame(CITY_LOCATIONS)


def station_hotspots() -> pd.DataFrame:
    """Return selected SAPS commercial-crime station hotspots."""
    frame = pd.DataFrame(SAPS_STATION_HOTSPOTS)
    frame["source"] = "SAPS Police Recorded Crime Statistics"
    frame["period"] = "Jan-Mar 2025"
    return frame
