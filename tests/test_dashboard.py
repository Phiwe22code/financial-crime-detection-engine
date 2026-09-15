"""Lightweight startup tests for the Dash command centre."""

import app


def test_dash_server_starts():
    response = app.server.test_client().get("/")
    assert response.status_code == 200
    assert b"Sentinel ZA" in response.data


def test_dashboard_registers_expected_callbacks():
    app.server.test_client().get("/")
    assert len(app.dashboard.callback_map) == 5


def test_all_four_views_render_with_project_data():
    if app.DATA is None:
        return
    assert app.executive_overview() is not None
    assert app.alert_queue_page() is not None
    assert app.account_page() is not None
    assert app.fraud_lab_page() is not None
