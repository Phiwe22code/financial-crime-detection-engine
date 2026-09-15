"""Dash command centre for the Financial Crime Detection Engine."""

from __future__ import annotations

from typing import Any

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, ctx, dcc, html, no_update

from src.dashboard_data import DashboardData, alert_queue, load_dashboard_data
from src.hotspots import SABRIC_SOURCE, SAPS_SOURCE, province_hotspots, station_hotspots
from src.simulator import RISK_COLORS, generate_event


APP_TITLE = "Sentinel ZA | Financial Crime Command Centre"
GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}

try:
    DATA: DashboardData | None = load_dashboard_data()
    DATA_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - exercised by a manual startup check
    DATA = None
    DATA_ERROR = str(exc)


dashboard = Dash(
    __name__,
    title=APP_TITLE,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
)
# Vercel's Python runtime discovers a WSGI application named ``app``.
# Dash itself remains available as ``dashboard`` for layouts and callbacks.
app = dashboard.server
server = app


def format_zar(value: float, compact: bool = False) -> str:
    """Format a number as South African rand."""
    if compact and abs(value) >= 1_000_000:
        return f"R {value / 1_000_000:,.1f}m"
    if compact and abs(value) >= 1_000:
        return f"R {value / 1_000:,.1f}k"
    return f"R {value:,.0f}"


def metric_card(
    label: str,
    value: str,
    detail: str = "",
    tone: str = "neutral",
) -> dbc.Col:
    """Create a consistent dashboard KPI card."""
    return dbc.Col(
        html.Div(
            [
                html.Div(label, className="metric-label"),
                html.Div(value, className="metric-value"),
                html.Div(detail, className="metric-detail"),
            ],
            className=f"metric-card metric-{tone}",
        ),
        xs=12,
        sm=6,
        xl=3,
    )


def page_heading(kicker: str, title: str, description: str) -> html.Div:
    return html.Div(
        [
            html.Div(kicker, className="page-kicker"),
            html.H1(title),
            html.P(description, className="page-description"),
        ],
        className="page-heading",
    )


def graph_card(title: str, subtitle: str, figure: go.Figure) -> html.Div:
    return html.Div(
        [
            html.Div([html.H3(title), html.P(subtitle)], className="card-heading"),
            dcc.Graph(figure=figure, config=GRAPH_CONFIG, className="dashboard-graph"),
        ],
        className="panel-card",
    )


def style_figure(figure: go.Figure, height: int = 350) -> go.Figure:
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#aebbd0", "family": "Inter, Segoe UI, sans-serif"},
        margin={"l": 38, "r": 20, "t": 20, "b": 38},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hoverlabel={"bgcolor": "#101827", "font_color": "#f8fafc"},
    )
    figure.update_xaxes(gridcolor="rgba(148,163,184,.10)", zeroline=False)
    figure.update_yaxes(gridcolor="rgba(148,163,184,.10)", zeroline=False)
    return figure


def empty_state(title: str, body: str) -> html.Div:
    return html.Div(
        [html.Div("—", className="empty-icon"), html.H3(title), html.P(body)],
        className="empty-state",
    )


def navigation() -> html.Aside:
    links = [
        ("OV", "Executive Overview", "/"),
        ("AQ", "High-Risk Alerts", "/alerts"),
        ("AD", "Account Deep Dive", "/account"),
        ("LIVE", "SA Live Fraud Lab", "/fraud-lab"),
    ]
    return html.Aside(
        [
            html.Div(
                [
                    html.Div("S", className="brand-mark"),
                    html.Div(
                        [
                            html.Div("SENTINEL ZA", className="brand-name"),
                            html.Small("FINANCIAL INTELLIGENCE"),
                        ]
                    ),
                ],
                className="brand-lockup",
            ),
            html.Div("COMMAND CENTRE", className="nav-section-label"),
            dbc.Nav(
                [
                    dbc.NavLink(
                        [html.Span(code, className="nav-code"), html.Span(label)],
                        href=href,
                        active="exact",
                    )
                    for code, label, href in links
                ],
                vertical=True,
                pills=True,
                className="side-nav",
            ),
            html.Div(
                [
                    html.Div(
                        [html.Span(className="status-dot"), "ANALYTICS ONLINE"],
                        className="system-status",
                    ),
                    html.Div("Isolation Forest · v1.0", className="system-caption"),
                ],
                className="sidebar-footer",
            ),
        ],
        className="sidebar",
    )


dashboard.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="selected-account-store", storage_type="session"),
        dcc.Store(
            id="simulation-store",
            storage_type="session",
            data={"running": False, "tick": 0, "events": []},
        ),
        dcc.Interval(id="simulation-interval", interval=1_400, disabled=True),
        navigation(),
        html.Main(
            [
                html.Div(
                    [
                        html.Div("FINANCIAL CRIME DETECTION ENGINE", className="topbar-title"),
                        html.Div(
                            [html.Span(className="live-dot"), "SECURE LOCAL ENVIRONMENT"],
                            className="topbar-status",
                        ),
                    ],
                    className="topbar",
                ),
                html.Div(id="page-content", className="page-content"),
            ],
            className="main-shell",
        ),
    ],
    className="app-shell",
)


def executive_overview() -> html.Div:
    assert DATA is not None
    transactions = DATA.transactions
    risk = DATA.risk
    weekly = (
        transactions.set_index("timestamp")["amount"]
        .resample("W")
        .sum()
        .reset_index()
    )
    time_fig = px.area(weekly, x="timestamp", y="amount")
    time_fig.update_traces(
        line={"color": "#4f8cff", "width": 2.5},
        fillcolor="rgba(79,140,255,.16)",
        hovertemplate="%{x|%d %b %Y}<br>Volume: R %{y:,.0f}<extra></extra>",
    )
    style_figure(time_fig)

    cross_border = transactions.groupby("is_international")["amount"].sum().reset_index()
    cross_border["exposure"] = cross_border["is_international"].map(
        {0: "Domestic", 1: "International"}
    )
    cross_fig = px.pie(
        cross_border,
        values="amount",
        names="exposure",
        hole=0.72,
        color="exposure",
        color_discrete_map={"Domestic": "#4f8cff", "International": "#f59e0b"},
    )
    cross_fig.update_traces(
        textinfo="percent",
        hovertemplate="%{label}<br>R %{value:,.0f}<extra></extra>",
    )
    style_figure(cross_fig)

    hist_fig = px.histogram(
        risk,
        x="risk_score",
        color="risk_band",
        nbins=28,
        category_orders={"risk_band": ["Low", "Medium", "High"]},
        color_discrete_map=RISK_COLORS,
    )
    hist_fig.update_layout(bargap=0.12)
    hist_fig.update_xaxes(title="Risk score")
    hist_fig.update_yaxes(title="Accounts")
    style_figure(hist_fig)

    band_counts = (
        risk["risk_band"].value_counts().rename_axis("risk_band").reset_index(name="accounts")
    )
    band_fig = px.bar(
        band_counts,
        x="risk_band",
        y="accounts",
        color="risk_band",
        color_discrete_map=RISK_COLORS,
        category_orders={"risk_band": ["Low", "Medium", "High"]},
    )
    band_fig.update_layout(showlegend=False)
    band_fig.update_xaxes(title="")
    band_fig.update_yaxes(title="Accounts")
    style_figure(band_fig)

    high_count = int((risk["risk_band"] == "High").sum())
    return html.Div(
        [
            page_heading(
                "NETWORK INTELLIGENCE",
                "Executive Overview",
                "A consolidated view of transaction velocity, exposure and account risk across the monitored network.",
            ),
            dbc.Row(
                [
                    metric_card("TRANSACTIONS", f"{len(transactions):,}", "12-month monitoring window"),
                    metric_card("MONITORED ACCOUNTS", f"{len(DATA.accounts):,}", "Behavioural profiles active"),
                    metric_card("HIGH-RISK ALERTS", f"{high_count:,}", "Immediate analyst attention", "danger"),
                    metric_card(
                        "TOTAL EXPOSURE",
                        format_zar(float(transactions["amount"].sum()), compact=True),
                        "Processed transaction value",
                        "accent",
                    ),
                ],
                className="g-3 metric-row",
            ),
            dbc.Row(
                [
                    dbc.Col(graph_card("Transaction Velocity", "Weekly value processed across the network", time_fig), lg=8),
                    dbc.Col(graph_card("Cross-Border Exposure", "Share of transaction value", cross_fig), lg=4),
                ],
                className="g-3 chart-row",
            ),
            dbc.Row(
                [
                    dbc.Col(graph_card("Risk Distribution", "Account anomaly scores converted to a 0–100 scale", hist_fig), lg=7),
                    dbc.Col(graph_card("Triage Mix", "Accounts by current investigation band", band_fig), lg=5),
                ],
                className="g-3 chart-row",
            ),
        ]
    )


def alert_queue_page() -> html.Div:
    assert DATA is not None
    queue = alert_queue(DATA)
    rows = queue[
        ["account_id", "account", "risk_score", "risk_band", "total_exposure", "explanation"]
    ].to_dict("records")
    columns = [
        {"field": "account", "headerName": "ACCOUNT", "width": 140, "pinned": "left"},
        {"field": "risk_score", "headerName": "RISK", "width": 110, "sort": "desc", "type": "numericColumn"},
        {"field": "risk_band", "headerName": "BAND", "width": 110},
        {
            "field": "total_exposure",
            "headerName": "EXPOSURE (ZAR)",
            "width": 170,
            "valueFormatter": {"function": "'R ' + d3.format(',.0f')(params.value)"},
        },
        {
            "field": "explanation",
            "headerName": "PRIMARY ANOMALY DRIVERS",
            "flex": 1,
            "minWidth": 420,
            "wrapText": True,
            "autoHeight": True,
        },
    ]
    grid = (
        dag.AgGrid(
            id="alerts-grid",
            rowData=rows,
            columnDefs=columns,
            defaultColDef={"filter": True, "sortable": True, "resizable": True},
            dashGridOptions={
                "rowSelection": {"mode": "singleRow", "enableClickSelection": True},
                "getRowId": {"function": "params.data.account_id"},
                "animateRows": True,
            },
            className="ag-theme-quartz-dark command-grid",
            style={"height": "590px", "width": "100%"},
        )
        if rows
        else empty_state("No high-risk accounts", "The current model run produced no alerts requiring review.")
    )
    top_score = f"{queue['risk_score'].max():.1f}" if not queue.empty else "—"
    return html.Div(
        [
            page_heading(
                "ANALYST WORKBENCH",
                "High-Risk Alert Queue",
                "Prioritised accounts requiring review. Select a row to open its forensic profile.",
            ),
            dbc.Row(
                [
                    metric_card("OPEN ALERTS", str(len(queue)), "Risk score of 70 or higher", "danger"),
                    metric_card("CRITICAL EXPOSURE", format_zar(float(queue["total_exposure"].sum()), compact=True), "Value linked to high-risk accounts", "accent"),
                    metric_card("TOP RISK SCORE", top_score, "Highest current anomaly score"),
                    metric_card("REVIEW STATUS", "OPEN", "Select any row to investigate"),
                ],
                className="g-3 metric-row",
            ),
            html.Div(
                [
                    html.Div([html.H3("Priority alerts"), html.P("Use the column headers to filter or sort the queue.")], className="card-heading table-heading"),
                    grid,
                ],
                className="panel-card table-card",
            ),
        ]
    )


def account_page(selected_account: str | None = None) -> html.Div:
    assert DATA is not None
    ordered = DATA.risk.sort_values("risk_score", ascending=False)
    options = [
        {
            "label": f"{row.account_id[:8]}…  ·  Risk {row.risk_score:.1f}  ·  {row.risk_band}",
            "value": row.account_id,
        }
        for row in ordered.itertuples()
    ]
    account_id = selected_account or options[0]["value"]
    return html.Div(
        [
            page_heading(
                "FORENSIC PROFILE",
                "Account Deep Dive",
                "Inspect behavioural signals, transaction history and the factors behind an account's current risk position.",
            ),
            html.Div(
                [
                    html.Label("ACCOUNT UNDER REVIEW", className="field-label"),
                    dcc.Dropdown(
                        id="account-selector",
                        options=options,
                        value=account_id,
                        clearable=False,
                        searchable=True,
                        className="account-select",
                    ),
                ],
                className="account-toolbar",
            ),
            dcc.Loading(html.Div(id="account-content"), type="circle", color="#4f8cff"),
        ]
    )


def _account_detail(account_id: str) -> html.Div:
    assert DATA is not None
    info = DATA.risk[DATA.risk["account_id"] == account_id].iloc[0]
    transactions = DATA.transactions[DATA.transactions["account_id"] == account_id].copy()
    account_features = DATA.features.loc[account_id]
    explanation_rows = DATA.explanations[DATA.explanations["account_id"] == account_id]
    explanation = (
        explanation_rows["explanation"].iloc[0]
        if not explanation_rows.empty
        else "No dominant statistical outlier was found. Continue monitoring the account's overall behaviour."
    )

    radar_columns = [
        "transaction_count",
        "avg_amount",
        "max_daily_spend",
        "cross_border_ratio",
        "night_ratio",
        "merchant_diversity",
    ]
    labels = ["Volume", "Average amount", "Daily peak", "Cross-border", "Night activity", "Merchants"]
    median = DATA.features[radar_columns].median().replace(0, 0.01)
    ratios = (account_features[radar_columns] / median).clip(0, 3)
    radar = go.Figure()
    radar.add_trace(
        go.Scatterpolar(
            r=[1] * len(labels),
            theta=labels,
            fill="toself",
            name="Network median",
            line_color="#2dd4bf",
            fillcolor="rgba(45,212,191,.10)",
        )
    )
    radar.add_trace(
        go.Scatterpolar(
            r=ratios.values,
            theta=labels,
            fill="toself",
            name="Selected account",
            line_color="#fb4b64",
            fillcolor="rgba(251,75,100,.20)",
        )
    )
    radar.update_layout(
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "radialaxis": {"visible": False, "range": [0, 3]},
            "angularaxis": {"gridcolor": "rgba(148,163,184,.18)"},
        }
    )
    style_figure(radar, 370)

    transactions = transactions.sort_values("timestamp")
    scope = transactions["is_international"].map({0: "Domestic", 1: "International"})
    timeline = px.scatter(
        transactions,
        x="timestamp",
        y="amount",
        color=scope,
        size="amount",
        hover_data=["merchant_id", "country", "transaction_type"],
        color_discrete_map={"Domestic": "#4f8cff", "International": "#f59e0b"},
    )
    timeline.update_xaxes(title="")
    timeline.update_yaxes(title="Amount (ZAR)")
    timeline.update_layout(legend_title_text="")
    style_figure(timeline, 370)

    log = transactions.sort_values("timestamp", ascending=False).head(250).copy()
    log["timestamp"] = log["timestamp"].dt.strftime("%d %b %Y  %H:%M")
    log["scope"] = log["is_international"].map({0: "Domestic", 1: "International"})
    log_rows = log[
        ["timestamp", "amount", "transaction_type", "channel", "country", "scope"]
    ].to_dict("records")
    log_columns = [
        {"field": "timestamp", "headerName": "TIMESTAMP", "width": 180},
        {
            "field": "amount",
            "headerName": "AMOUNT (ZAR)",
            "width": 150,
            "valueFormatter": {"function": "'R ' + d3.format(',.2f')(params.value)"},
        },
        {"field": "transaction_type", "headerName": "TYPE", "width": 140},
        {"field": "channel", "headerName": "CHANNEL", "width": 130},
        {"field": "country", "headerName": "COUNTRY", "width": 120},
        {"field": "scope", "headerName": "SCOPE", "flex": 1},
    ]
    tone = str(info["risk_band"]).lower()
    return html.Div(
        [
            dbc.Row(
                [
                    metric_card("ACCOUNT", f"{account_id[:8]}…", f"Customer {str(info['customer_id'])[:8]}…"),
                    metric_card("RISK SCORE", f"{info['risk_score']:.1f}", f"{info['risk_band']} risk band", "danger" if tone == "high" else "accent"),
                    metric_card("TRANSACTIONS", f"{len(transactions):,}", "Recorded account activity"),
                    metric_card("TOTAL SPEND", format_zar(float(transactions["amount"].sum()), compact=True), "Cumulative exposure"),
                ],
                className="g-3 metric-row",
            ),
            html.Div(
                [
                    html.Div([html.Span(info["risk_band"], className=f"risk-pill risk-{tone}"), html.Strong(" AUTOMATED FORENSICS")]),
                    html.P(explanation),
                ],
                className=f"forensic-banner forensic-{tone}",
            ),
            dbc.Row(
                [
                    dbc.Col(graph_card("Behavioural Profile", "Selected account versus the network median", radar), lg=5),
                    dbc.Col(graph_card("Transaction Timeline", "Large and international movements across time", timeline), lg=7),
                ],
                className="g-3 chart-row",
            ),
            html.Div(
                [
                    html.Div([html.H3("Forensic Transaction Log"), html.P("Most recent 250 account transactions")], className="card-heading table-heading"),
                    dag.AgGrid(
                        rowData=log_rows,
                        columnDefs=log_columns,
                        defaultColDef={"sortable": True, "filter": True, "resizable": True},
                        dashGridOptions={"animateRows": True},
                        className="ag-theme-quartz-dark command-grid",
                        style={"height": "430px", "width": "100%"},
                    ),
                ],
                className="panel-card table-card",
            ),
        ]
    )


def hotspot_map(events: list[dict[str, Any]]) -> go.Figure:
    provinces = province_hotspots()
    stations = station_hotspots()
    figure = go.Figure()
    figure.add_trace(
        go.Scattermap(
            lat=provinces["lat"],
            lon=provinces["lon"],
            mode="markers+text",
            text=provinces["province"],
            textposition="bottom center",
            customdata=provinces[["province", "credit_share", "debit_share", "period"]],
            marker={
                "size": 18 + provinces["credit_share"] * 0.75,
                "color": provinces["credit_share"],
                "colorscale": [[0, "#3b82f6"], [0.45, "#f59e0b"], [1, "#fb4b64"]],
                "cmin": 0,
                "cmax": 50,
                "opacity": 0.52,
                "showscale": True,
                "colorbar": {"title": "Credit fraud %", "thickness": 10, "len": 0.58, "x": 1.01},
            },
            name="SABRIC province share",
            hovertemplate="<b>%{customdata[0]}</b><br>Credit-card fraud share: %{customdata[1]:.1f}%<br>Debit-card fraud share: %{customdata[2]:.1f}%<br>Period: %{customdata[3]}<extra>SABRIC</extra>",
        )
    )
    figure.add_trace(
        go.Scattermap(
            lat=stations["lat"],
            lon=stations["lon"],
            mode="markers",
            customdata=stations[["station", "district", "province", "cases", "period"]],
            marker={"size": 10, "color": "#e2e8f0", "symbol": "diamond", "opacity": 0.8},
            name="SAPS commercial-crime station",
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}, %{customdata[2]}<br>Reported commercial-crime cases: %{customdata[3]}<br>Period: %{customdata[4]}<extra>SAPS</extra>",
        )
    )
    if events:
        frame = pd.DataFrame(events)
        for band in ["Low", "Medium", "High"]:
            subset = frame[frame["risk_band"] == band]
            if subset.empty:
                continue
            figure.add_trace(
                go.Scattermap(
                    lat=subset["lat"],
                    lon=subset["lon"],
                    mode="markers",
                    customdata=subset[
                        ["transaction_id", "city", "province", "amount", "risk_score", "scenario_label"]
                    ],
                    marker={
                        "size": 9 if band == "Low" else 13 if band == "Medium" else 18,
                        "color": RISK_COLORS[band],
                        "opacity": 0.9,
                    },
                    name=f"Simulated · {band}",
                    hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}, %{customdata[2]}<br>Amount: R %{customdata[3]:,.2f}<br>Risk: %{customdata[4]:.1f}<br>%{customdata[5]}<extra>Simulation</extra>",
                )
            )
    figure.update_layout(
        height=600,
        map={"style": "carto-darkmatter", "center": {"lat": -29.0, "lon": 25.2}, "zoom": 4.2},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cbd5e1"},
        legend={"orientation": "h", "x": 0.01, "y": 0.01, "bgcolor": "rgba(8,15,27,.82)"},
        uirevision="south-africa-command-map",
    )
    return figure


def fraud_lab_page() -> html.Div:
    return html.Div(
        [
            page_heading(
                "REAL-TIME SIMULATION",
                "SA Live Fraud Lab",
                "Start an automatic stream of South African transactions, score emerging account behaviour and watch alerts appear against real historical hotspot context.",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Button([html.Span("▶"), " START SIMULATION"], id="start-simulation", n_clicks=0, className="control-btn control-primary"),
                            html.Button([html.Span("Ⅱ"), " PAUSE"], id="pause-simulation", n_clicks=0, className="control-btn"),
                            html.Button([html.Span("↺"), " RESET"], id="reset-simulation", n_clicks=0, className="control-btn"),
                        ],
                        className="simulation-controls",
                    ),
                    html.Div(id="simulation-status", className="simulation-status status-ready"),
                ],
                className="lab-toolbar",
            ),
            dbc.Row(id="simulation-kpis", className="g-3 metric-row"),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div([html.H3("South African Risk Map"), html.P("Historical banking-fraud context with a separate live simulation overlay")], className="card-heading"),
                                        html.Div([html.Span(className="legend-dot historical"), "Historical", html.Span(className="legend-dot simulated"), "Simulated"], className="map-layer-key"),
                                    ],
                                    className="map-card-header",
                                ),
                                dcc.Graph(id="hotspot-map", figure=hotspot_map([]), config=GRAPH_CONFIG, className="hotspot-map"),
                                html.Div(
                                    [
                                        html.Span("SOURCE NOTE"),
                                        html.A("SABRIC 2024", href=SABRIC_SOURCE, target="_blank"),
                                        html.A("SAPS Jan–Mar 2025", href=SAPS_SOURCE, target="_blank"),
                                        html.Em("Historical geography is context only and does not determine individual risk."),
                                    ],
                                    className="source-note",
                                ),
                            ],
                            className="panel-card map-card",
                        ),
                        xl=8,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Div([html.H3("Live Alert Stream"), html.P("High-risk events requiring attention")], className="card-heading"),
                                dag.AgGrid(
                                    id="lab-alert-grid",
                                    rowData=[],
                                    columnDefs=[
                                        {"field": "time", "headerName": "TIME", "width": 90},
                                        {"field": "account", "headerName": "ACCOUNT", "width": 115},
                                        {"field": "city", "headerName": "CITY", "width": 120},
                                        {"field": "risk_score", "headerName": "RISK", "width": 85},
                                        {"field": "scenario_label", "headerName": "SIGNAL", "flex": 1, "minWidth": 175},
                                    ],
                                    defaultColDef={"sortable": True, "resizable": True},
                                    dashGridOptions={
                                        "rowSelection": {"mode": "singleRow", "enableClickSelection": True},
                                        "getRowId": {"function": "params.data.transaction_id"},
                                    },
                                    className="ag-theme-quartz-dark command-grid lab-alert-grid",
                                    style={"height": "585px", "width": "100%"},
                                ),
                            ],
                            className="panel-card table-card lab-alert-card",
                        ),
                        xl=4,
                    ),
                ],
                className="g-3 chart-row",
            ),
            html.Div(
                [
                    html.Div([html.H3("Transaction Feed"), html.P("Latest automatically generated simulation events")], className="card-heading table-heading"),
                    dag.AgGrid(
                        id="live-feed-grid",
                        rowData=[],
                        columnDefs=[
                            {"field": "time", "headerName": "TIME", "width": 100},
                            {"field": "account", "headerName": "ACCOUNT", "width": 125},
                            {"field": "location", "headerName": "LOCATION", "width": 220},
                            {"field": "amount", "headerName": "AMOUNT", "width": 140, "valueFormatter": {"function": "'R ' + d3.format(',.2f')(params.value)"}},
                            {"field": "channel", "headerName": "CHANNEL", "width": 120},
                            {"field": "scenario_label", "headerName": "BEHAVIOUR", "flex": 1, "minWidth": 210},
                            {"field": "risk_score", "headerName": "RISK", "width": 95},
                            {"field": "risk_band", "headerName": "BAND", "width": 105},
                        ],
                        defaultColDef={"sortable": True, "resizable": True},
                        dashGridOptions={"animateRows": True, "getRowId": {"function": "params.data.transaction_id"}},
                        className="ag-theme-quartz-dark command-grid",
                        style={"height": "380px", "width": "100%"},
                    ),
                ],
                className="panel-card table-card",
            ),
        ]
    )


def error_page() -> html.Div:
    return html.Div(
        [
            page_heading("SYSTEM NOTICE", "Analytics engine unavailable", "The dashboard could not initialise its local data pipeline."),
            html.Div(
                [html.H3("Database connection failed"), html.P(DATA_ERROR or "Unknown error"), html.Code("python -m src.data_generation")],
                className="error-panel",
            ),
        ]
    )


@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    State("selected-account-store", "data"),
)
def render_page(pathname: str, selected_account: str | None) -> html.Div:
    if DATA is None:
        return error_page()
    if pathname == "/alerts":
        return alert_queue_page()
    if pathname == "/account":
        return account_page(selected_account)
    if pathname == "/fraud-lab":
        return fraud_lab_page()
    return executive_overview()


@callback(Output("account-content", "children"), Input("account-selector", "value"))
def update_account_detail(account_id: str | None) -> html.Div:
    if not account_id or DATA is None:
        return empty_state("Select an account", "Choose an account to open its forensic profile.")
    return _account_detail(account_id)


@callback(
    Output("selected-account-store", "data"),
    Output("url", "pathname"),
    Input("alerts-grid", "selectedRows", allow_optional=True),
    Input("lab-alert-grid", "selectedRows", allow_optional=True),
    prevent_initial_call=True,
)
def open_selected_account(
    queue_rows: list[dict[str, Any]] | None,
    lab_rows: list[dict[str, Any]] | None,
) -> tuple[str, str] | tuple[Any, Any]:
    rows = queue_rows if ctx.triggered_id == "alerts-grid" else lab_rows
    if not rows:
        return no_update, no_update
    return rows[0]["account_id"], "/account"


@callback(
    Output("simulation-store", "data"),
    Output("simulation-interval", "disabled"),
    Input("start-simulation", "n_clicks", allow_optional=True),
    Input("pause-simulation", "n_clicks", allow_optional=True),
    Input("reset-simulation", "n_clicks", allow_optional=True),
    Input("simulation-interval", "n_intervals"),
    State("simulation-store", "data"),
    prevent_initial_call=True,
)
def control_simulation(
    _start: int | None,
    _pause: int | None,
    _reset: int | None,
    _interval: int,
    state: dict[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    state = state or {"running": False, "tick": 0, "events": []}
    trigger = ctx.triggered_id
    if trigger == "start-simulation" and _start:
        return {**state, "running": True}, False
    if trigger == "pause-simulation" and _pause:
        return {**state, "running": False}, True
    if trigger == "reset-simulation" and _reset:
        return {"running": False, "tick": 0, "events": []}, True
    if trigger == "simulation-interval" and state.get("running") and DATA is not None:
        events = list(state.get("events", []))
        tick = int(state.get("tick", 0)) + 1
        event = generate_event(DATA, tick=tick, prior_events=events)
        events.append(event)
        return {"running": True, "tick": tick, "events": events[-180:]}, False
    return state, not bool(state.get("running"))


@callback(
    Output("simulation-kpis", "children"),
    Output("hotspot-map", "figure"),
    Output("live-feed-grid", "rowData"),
    Output("lab-alert-grid", "rowData"),
    Output("simulation-status", "children"),
    Output("simulation-status", "className"),
    Input("simulation-store", "data"),
)
def update_fraud_lab(state: dict[str, Any] | None):
    state = state or {"running": False, "tick": 0, "events": []}
    events = list(state.get("events", []))
    alerts = [event for event in events if event.get("is_alert")]
    total_value = sum(float(event["amount"]) for event in events)
    max_risk = max((float(event["risk_score"]) for event in events), default=0.0)
    top_province = "—"
    if events:
        top_province = pd.Series([event["province"] for event in events]).value_counts().index[0]

    kpis = [
        metric_card("PROCESSED", f"{len(events):,}", "Session transactions"),
        metric_card("LIVE ALERTS", f"{len(alerts):,}", "High-risk simulated events", "danger"),
        metric_card("HIGHEST RISK", f"{max_risk:.1f}", "Peak session score", "accent"),
        metric_card("SIMULATED VALUE", format_zar(total_value, compact=True), f"Most active: {top_province}"),
    ]

    def display_row(event: dict[str, Any]) -> dict[str, Any]:
        return {
            **event,
            "time": pd.Timestamp(event["timestamp"]).strftime("%H:%M:%S"),
            "account": event["account_id"][:8] + "…",
            "location": f"{event['city']}, {event['province']}",
        }

    feed_rows = [display_row(event) for event in reversed(events[-40:])]
    alert_rows = [display_row(event) for event in reversed(alerts[-30:])]
    running = bool(state.get("running"))
    status = "STREAMING · AUTOMATIC MONITORING ACTIVE" if running else "READY · CLICK START SIMULATION"
    class_name = "simulation-status status-running" if running else "simulation-status status-ready"
    return kpis, hotspot_map(events), feed_rows, alert_rows, status, class_name


if __name__ == "__main__":
    dashboard.run(host="127.0.0.1", port=8050, debug=False)
