"""
ui/hospital_dashboard.py
🏥 Fun Q-Hospital Command Center — Beige & Fun Theme.
Quantum-Assisted Bed Allocation with PQC Security.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import json
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from pipeline import run_pipeline
from data.generator import RESOURCE_CAPACITY

# ── Custom CSS & Theme ────────────────────────────────────────────────────────

CUSTOM_STYLE = """
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
    --bg-beige: #fdfaf5;
    --card-white: #ffffff;
    --accent-red: #ff6b6b;
    --accent-teal: #1abc9c;
    --accent-yellow: #f1c40f;
    --text-dark: #2d3436;
    --border-ink: #2d3436;
}

body {
    background: var(--bg-beige) !important;
    font-family: 'Fredoka', sans-serif !important;
    color: var(--text-dark) !important;
    overflow-x: hidden;
}

.command-card {
    background: var(--card-white) !important;
    border: 3px solid var(--border-ink) !important;
    border-radius: 16px !important;
    box-shadow: 6px 6px 0px var(--border-ink) !important;
    transition: all 0.1s ease;
}

.command-card:hover {
    transform: translate(-2px, -2px);
    box-shadow: 10px 10px 0px var(--border-ink) !important;
}

.bed-card {
    background: white !important;
    border: 3px solid var(--border-ink) !important;
    border-radius: 14px !important;
    padding: 12px;
    margin-bottom: 12px;
    box-shadow: 4px 4px 0px var(--border-ink) !important;
}

.header-blur {
    background: #ffffff !important;
    border-bottom: 4px solid var(--border-ink) !important;
    box-shadow: 0 4px 0 rgba(0,0,0,0.05);
}

.metric-value {
    font-family: 'Fredoka', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-dark) !important;
}

.run-button {
    background: #ff6b6b !important;
    border: 3px solid var(--border-ink) !important;
    border-radius: 14px !important;
    box-shadow: 6px 6px 0px var(--border-ink) !important;
    color: white !important;
    font-weight: 700 !important;
    text-transform: uppercase;
}

.run-button:active {
    transform: translate(2px, 2px);
    box-shadow: 2px 2px 0px var(--border-ink) !important;
}

.table {
    border: 3px solid var(--border-ink) !important;
    background: white !important;
    border-radius: 12px !important;
    overflow: hidden;
}

.table th {
    background: #2d3436 !important;
    color: white !important;
    border-bottom: none !important;
}

.empty-bed {
    border: 3px dashed #b2bec3;
    background: #fdfaf5;
    border-radius: 14px;
    height: 100px;
    margin-bottom: 15px;
}

.pulse-critical {
    border-color: #ff6b6b !important;
    animation: bounce 0.6s infinite alternate;
}

@keyframes bounce {
    from { transform: rotate(-1deg) scale(1); }
    to { transform: rotate(1deg) scale(1.02); }
}
"""

# ── UI Helpers ────────────────────────────────────────────────────────────────

def metric_card(title, value, subtitle="", color="#ff6b6b", icon=""):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div([
            html.Span(icon, style={"fontSize": "1.5rem", "marginRight": "10px"}),
            html.Span(title, style={"color": "#636e72", "fontSize": "0.85rem", "textTransform": "uppercase", "fontWeight": "700"}),
        ], className="d-flex align-items-center mb-2"),
        html.Div(str(value), className="metric-value", style={
            "fontSize": "2.2rem", "lineHeight": "1", "color": color
        }),
        html.Div(subtitle, style={"color": "#636e72", "fontSize": "0.8rem", "marginTop": "8px", "fontWeight": "500"}),
    ]), className="command-card"), width=3)

def bed_card(patient_id, urgency, bp, spo2):
    is_critical = urgency >= 0.7
    pulse_class = "pulse-critical" if is_critical else ""
    
    return html.Div([
        html.Div([
            html.Div([
                html.Small("PATIENT ID", style={"fontSize": "0.6rem", "color": "#636e72", "display": "block", "fontWeight": "700"}),
                html.Div(patient_id, style={"fontWeight": "700", "fontSize": "1.1rem"}),
            ]),
            html.Div([
                html.Small("URGENCY", style={"fontSize": "0.6rem", "color": "#636e72", "display": "block", "fontWeight": "700", "textAlign": "right"}),
                html.Div(f"{urgency:.2f}", style={"fontWeight": "700", "color": "#ff6b6b" if is_critical else "#2d3436"}),
            ])
        ], className="d-flex justify-content-between align-items-start mb-2"),
        
        html.Div([
            html.Div([
                html.Small("BP DEV", style={"fontSize": "0.55rem", "color": "#b2bec3", "fontWeight": "700"}),
                html.Div(f"{bp:.2f}", style={"fontSize": "0.8rem", "fontWeight": "600"}),
            ], style={"marginRight": "15px"}),
            html.Div([
                html.Small("O₂ SAT", style={"fontSize": "0.55rem", "color": "#b2bec3", "fontWeight": "700"}),
                html.Div(f"{1.0-spo2:.2f}", style={"fontSize": "0.8rem", "fontWeight": "600"}),
            ])
        ], className="d-flex")
    ], className=f"bed-card {pulse_class}")

def resource_column(title, icon, color, capacity, assigned, df):
    used = len(assigned)
    cards = []
    for a in assigned:
        idx  = a["patient_idx"]
        pid  = df.loc[idx, "patient_id"] if idx < len(df) else f"P{idx}"
        bp   = float(df.loc[idx, "bp_deviation"])  if idx < len(df) else 0.0
        spo2 = float(df.loc[idx, "spo2_deficit"])  if idx < len(df) else 0.0
        cards.append(bed_card(pid, a["urgency"], bp, spo2))
    
    for _ in range(capacity - used):
        cards.append(html.Div([
            html.Div("FREE BED", style={"fontSize": "0.7rem", "fontWeight": "700", "color": "#b2bec3"})
        ], className="empty-bed d-flex align-items-center justify-content-center"))

    return dbc.Col([
        html.Div([
            html.Div([
                html.Span(icon, style={"fontSize": "1.3rem", "marginRight": "10px"}),
                html.Span(title, style={"fontWeight": "700", "fontSize": "1rem", "color": "white"}),
            ]),
            dbc.Badge(f"{used}/{capacity} FULL", color="light", text_color="dark",
                      style={"borderRadius": "8px", "fontWeight": "700"})
        ], style={
            "background": "#2d3436",
            "border": "3px solid #2d3436",
            "borderRadius": "14px 14px 0 0",
            "padding": "12px 18px",
            "marginBottom": "0px",
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center"
        }),
        html.Div(cards, style={
            "background": "white",
            "border": "3px solid #2d3436",
            "borderTop": "none",
            "borderRadius": "0 0 14px 14px",
            "padding": "18px",
            "boxShadow": "6px 6px 0px rgba(0,0,0,0.05)"
        }),
    ], width=4)


# ── Custom CSS & Theme ────────────────────────────────────────────────────────

CUSTOM_STYLE = """
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
    --bg-beige: #fdfaf5;
    --card-white: #ffffff;
    --accent-red: #ff6b6b;
    --accent-teal: #1abc9c;
    --accent-yellow: #f1c40f;
    --text-dark: #2d3436;
    --border-ink: #2d3436;
}

body {
    background: var(--bg-beige) !important;
    font-family: 'Fredoka', sans-serif !important;
    color: var(--text-dark) !important;
    overflow-x: hidden;
}

.command-card {
    background: var(--card-white) !important;
    border: 3px solid var(--border-ink) !important;
    border-radius: 16px !important;
    box-shadow: 6px 6px 0px var(--border-ink) !important;
    transition: all 0.1s ease;
}

.command-card:hover {
    transform: translate(-2px, -2px);
    box-shadow: 10px 10px 0px var(--border-ink) !important;
}

.bed-card {
    background: white !important;
    border: 3px solid var(--border-ink) !important;
    border-radius: 14px !important;
    padding: 12px;
    margin-bottom: 12px;
    box-shadow: 4px 4px 0px var(--border-ink) !important;
}

.header-blur {
    background: #ffffff !important;
    border-bottom: 4px solid var(--border-ink) !important;
    box-shadow: 0 4px 0 rgba(0,0,0,0.05);
}

.metric-value {
    font-family: 'Fredoka', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-dark) !important;
}

.run-button {
    background: #ff6b6b !important;
    border: 3px solid var(--border-ink) !important;
    border-radius: 14px !important;
    box-shadow: 6px 6px 0px var(--border-ink) !important;
    color: white !important;
    font-weight: 700 !important;
    text-transform: uppercase;
}

.run-button:active {
    transform: translate(2px, 2px);
    box-shadow: 2px 2px 0px var(--border-ink) !important;
}

.table {
    border: 3px solid var(--border-ink) !important;
    background: white !important;
    border-radius: 12px !important;
    overflow: hidden;
}

.table th {
    background: #2d3436 !important;
    color: white !important;
    border-bottom: none !important;
}

.empty-bed {
    border: 3px dashed #b2bec3;
    background: #fdfaf5;
    border-radius: 14px;
    height: 100px;
    margin-bottom: 15px;
}

.pulse-critical {
    border-color: #ff6b6b !important;
    animation: bounce 0.6s infinite alternate;
}

@keyframes bounce {
    from { transform: rotate(-1deg) scale(1); }
    to { transform: rotate(1deg) scale(1.02); }
}
"""

import datetime

# ── Metric cards helper ───────────────────────────────────────────────────

def metric_card(title, value, subtitle="", color="#ff6b6b", icon=""):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div([
            html.Span(icon, style={"fontSize": "1.5rem", "marginRight": "10px"}),
            html.Span(title, style={"color": "#636e72", "fontSize": "0.85rem", "textTransform": "uppercase", "fontWeight": "700"}),
        ], className="d-flex align-items-center mb-2"),
        html.Div(str(value), className="metric-value", style={
            "fontSize": "2.2rem", "lineHeight": "1", "color": color
        }),
        html.Div(subtitle, style={"color": "#636e72", "fontSize": "0.8rem", "marginTop": "8px", "fontWeight": "500"}),
    ]), className="command-card"), width=3)


# ── App layout ────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600;700&display=swap"
    ],
    title="🏥 Fun Q-Hospital",
)
app.index_string = app.index_string.replace(
    "</head>",
    f"<style>{CUSTOM_STYLE}</style></head>",
)

app.layout = html.Div([
    dcc.Store(id="pipeline-store"),

    # ── Top nav ──────────────────────────────────────────────────────────
    html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H3("🏥  Fun Q-Hospital",
                                style={"margin": 0, "color": "#2d3436",
                                       "fontWeight": "700"}),
                        html.Small("Quantum-Powered Bed Allocation",
                                   style={"color": "#636e72", "fontWeight": "600"}),
                    ]),
                ], width=5),
                dbc.Col([
                    html.Div([
                        dbc.Badge("🔒 SECURE", color="dark", className="px-3 py-2 me-2",
                                  style={"borderRadius": "12px", "fontSize": "0.75rem", "fontWeight": "700"}),
                        dbc.Badge("PQC ACTIVE", color="danger", className="px-3 py-2 me-2",
                                  style={"borderRadius": "12px", "fontSize": "0.75rem"}),
                        html.Div(id="last-update", style={"color": "#2d3436", "fontSize": "0.8rem", "fontWeight": "700"})
                    ], className="d-flex align-items-center justify-content-end")
                ], width=7),
            ], align="center"),
        ], fluid=True),
    ], className="header-blur py-3 mb-4"),

    dbc.Container([

        # ── Controls ─────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("🌫️  SMOG INTENSITY (PM2.5)",
                                           style={"color": "#2d3436", "fontSize": "0.85rem", "fontWeight": "700"}),
                                dcc.Slider(id="aqi-slider", min=0, max=500, step=10, value=50,
                                           marks={0: "CLEAR", 250: "MID", 500: "SURGE"},
                                           tooltip={"placement": "bottom", "always_visible": False},
                                           className="mt-2"),
                            ], width=5),
                            dbc.Col([
                                html.Label("👤  PATIENT INTAKE",
                                           style={"color": "#2d3436", "fontSize": "0.85rem", "fontWeight": "700"}),
                                dcc.Slider(id="patients-slider", min=4, max=12, step=2, value=8,
                                           marks={4: "4", 8: "8", 12: "MAX"},
                                           className="mt-2"),
                            ], width=4),
                            dbc.Col([
                                dbc.Button("OPTIMIZE NOW! 🚀", id="run-btn", className="run-button w-100 py-2",
                                           style={"marginTop": "1.4rem"}),
                            ], width=3),
                        ])
                    ])
                ], className="command-card mb-4")
            ], width=12),
        ]),

        # ── Metric cards ──────────────────────────────────────────────────
        dbc.Row(id="metric-cards", className="mb-4"),

        # ── Bed grid ──────────────────────────────────────────────────────
        html.H6("🏨 LIVE WARD SNAPSHOT",
                style={"color": "#636e72", "fontSize": "0.85rem", "fontWeight": "700", "marginBottom": "1rem"}),
        dcc.Loading(
            id="loading-beds",
            type="cube",
            color="#ff6b6b",
            children=html.Div(id="bed-grid"),
        ),

        html.Hr(style={"borderColor": "#2d3436", "borderWidth": "3px", "margin": "30px 0"}),

        # ── Patient intake table ──────────────────────────────────────────
        html.H6("📋  PATIENT QUEUE",
                style={"color": "#636e72", "fontWeight": "700", "fontSize": "0.85rem"}),
        html.Div(id="patient-table"),

        html.Hr(style={"borderColor": "#2d3436", "borderWidth": "3px", "margin": "30px 0"}),

        # ── Security panel ────────────────────────────────────────────────
        html.H6("🛡️  QUANTUM SHIELD ACTIVATED",
                style={"color": "#636e72", "fontWeight": "700", "fontSize": "0.85rem"}),
        html.Div(id="security-panel"),

        html.Div(style={"height": "60px"}),

    ], fluid=True),

], style={"background": "#fdfaf5", "minHeight": "100vh"})


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("pipeline-store", "data"),
    Input("run-btn", "n_clicks"),
    State("aqi-slider", "value"),
    State("patients-slider", "value"),
    prevent_initial_call=True,
)
def run_and_store(_, aqi, n_patients):
    results = run_pipeline(aqi_level=float(aqi), n_patients=int(n_patients), verbose=False)
    return {
        "df":               results["df"].to_json(orient="split"),
        "q_alloc":          results["quantum_allocation"],
        "c_alloc":          results["classical_allocation"],
        "urgency_scores":   results["urgency_scores"].tolist(),
        "q_util":           results["quantum_utilization"],
        "c_util":           results["classical_utilization"],
        "unalloc_q":        results["unallocated_quantum"],
        "unalloc_c":        results["unallocated_classical"],
        "security":         results["security"],
        "encrypted_sample": results["encrypted_sample"],
        "audit_hash":       results["audit_hash"],
        "aqi":              aqi,
        "n_patients":       n_patients,
    }


@app.callback(
    Output("metric-cards", "children"),
    Output("bed-grid", "children"),
    Output("patient-table", "children"),
    Output("security-panel", "children"),
    Output("last-update", "children"),
    Input("pipeline-store", "data"),
)
def update_ui(data):
    if data is None:
        placeholder = dbc.Alert([
            html.H4("READY FOR ACTION!", className="alert-heading", style={"fontWeight": "700"}),
            html.P("Set your parameters and hit the big red button to start the simulation."),
        ], color="warning", className="text-center py-5", style={
            "border": "3px solid #2d3436", "borderRadius": "16px", "boxShadow": "8px 8px 0px #2d3436"
        })
        return [], placeholder, [], [], ""

    df       = pd.read_json(io.StringIO(data["df"]), orient="split")
    q_alloc  = data["q_alloc"]
    sec      = data["security"]
    aqi      = data["aqi"]
    n        = data["n_patients"]

    # ── Metric cards ─────────────────────────────────────────────────────
    urgencies = [a["urgency"] for a in q_alloc]
    n_critical = sum(1 for u in urgencies if u >= 0.7)
    
    cards = [
        metric_card("Live Queue", f"{n:02d}", "Patients being triaged", "#2d3436", "👤"),
        metric_card("Crisis Alert", f"{n_critical}", "Critical patients found", "#ff6b6b", "🚨"),
        metric_card("Smog Level", f"{aqi}", "Local PM2.5 surge weight", "#f1c40f", "🌫️"),
        metric_card("Quantum PQC", "SECURE", "Post-Quantum Protected", "#1abc9c", "🛡️"),
    ]

    # ── Bed grid ─────────────────────────────────────────────────────────
    resource_cfg = [
        (0, "ICU / Trauma",    "🏥", "#ff6b6b", RESOURCE_CAPACITY[0]),
        (1, "Vent Unit",       "🫁", "#f1c40f", RESOURCE_CAPACITY[1]),
        (2, "General Ward",    "🛏️",  "#1abc9c", RESOURCE_CAPACITY[2]),
    ]
    cols = []
    for ridx, title, icon, color, cap in resource_cfg:
        assigned = [a for a in q_alloc if a.get("resource_idx") == ridx]
        cols.append(resource_column(title, icon, color, cap, assigned, df))

    bed_grid = dbc.Row(cols, className="mb-4")

    # ── Patient intake table ──────────────────────────────────────────────
    rows = []
    for _, row in df.iterrows():
        u = float(row["urgency_score"])
        is_critical = u >= 0.7
        q_res = next(
            (a["resource_name"] for a in q_alloc if a["patient_idx"] == row.name),
            "⚠️ UNALLOCATED",
        )
        rows.append(html.Tr([
            html.Td(row["patient_id"],          style={"fontWeight": "700", "fontFamily": "JetBrains Mono"}),
            html.Td(f"{row['bp_deviation']:.2f}"),
            html.Td(f"{1.0-row['spo2_deficit']:.2f}"),
            html.Td(f"{row['aqi_pm25']:.1f}"),
            html.Td(f"{u:.3f}", style={"color": "#ff6b6b" if is_critical else "#2d3436", "fontWeight": "700"}),
            html.Td(dbc.Badge("URGENT" if is_critical else "STABLE", 
                              color="danger" if is_critical else "success",
                              style={"borderRadius": "8px"})),
            html.Td(q_res, style={"fontWeight": "700", "color": "#1abc9c"}),
        ]))

    table = dbc.Table(
        [html.Thead(html.Tr([
            html.Th("PATIENT"), html.Th("BP Δ"), html.Th("O₂ SAT"),
            html.Th("AQI"), html.Th("SCORE"), html.Th("STATUS"),
            html.Th("ASSIGNED TO"),
        ])),
         html.Tbody(rows)],
        bordered=False, hover=True, responsive=True, className="table align-middle",
    )

    # ── Security panel ────────────────────────────────────────────────────
    audit = data.get("audit_hash", "")[:48] + "..."
    sec_panel = dbc.Row([
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.Div([
                    dbc.Badge("PQC ACTIVE", color="danger", className="me-2", style={"borderRadius": "8px"}),
                    html.Span(sec["algorithm"], style={"fontWeight": "700", "fontSize": "0.9rem"}),
                ], className="mb-2"),
                html.Small(f"PUBKEY: {sec['pubkey_fp']}", 
                           style={"fontFamily": "JetBrains Mono", "display": "block", "color": "#636e72"}),
                html.Small(f"TARGET: {sec['nist_target']}", 
                           style={"fontWeight": "600", "color": "#1abc9c"}),
            ]), className="command-card"),
        ], width=6),
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.Small("AUDIT PROOF (QUANTUM LEDGER)", style={"fontWeight": "700", "color": "#636e72"}),
                html.Pre(audit, style={
                    "fontSize": "0.7rem", "margin": "8px 0 0 0",
                    "background": "#fdfaf5", "padding": "10px", "borderRadius": "10px",
                    "border": "2px solid #2d3436"
                }),
            ]), className="command-card"),
        ], width=6),
    ])

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    return cards, bed_grid, table, sec_panel, f"LIVE UPDATED: {ts}"


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, port=8050, host="0.0.0.0")
