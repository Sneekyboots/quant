"""
ui/hospital_dashboard.py
🏥 Fun Q-Hospital Command Center — Beige & Fun Theme.
Quantum-Assisted Bed Allocation with PQC Security.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import json
import time
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from pipeline import run_pipeline
from data.generator import RESOURCE_CAPACITY, STAFF_ROLES

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

.staff-chip {
    border-radius: 10px;
    padding: 6px 10px;
    margin-bottom: 6px;
    border: 2px solid #2d3436;
    box-shadow: 3px 3px 0px #2d3436;
    font-size: 0.78rem;
    background: white;
}

.fatigue-high  { border-left: 6px solid #ff6b6b !important; }
.fatigue-mid   { border-left: 6px solid #f1c40f !important; }
.fatigue-low   { border-left: 6px solid #1abc9c !important; }

.waitlist-card {
    background: #fff5f5;
    border: 3px solid #ff6b6b;
    border-left: 8px solid #ff6b6b;
    border-radius: 14px;
    padding: 12px 16px;
    margin-bottom: 10px;
    box-shadow: 4px 4px 0px #2d3436;
}

.fallback-badge {
    font-size: 0.6rem;
    background: #f1c40f;
    color: #2d3436;
    border: 2px solid #2d3436;
    border-radius: 6px;
    padding: 1px 6px;
    margin-left: 6px;
    font-weight: 700;
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

def bed_card(patient_id, urgency, bp, spo2, hr=0.0, gcs=0.0, lactate=0.0, reason="", is_fallback=False):
    is_critical = urgency >= 0.7
    pulse_class = "pulse-critical" if is_critical else ""

    gcs_color     = "#ff6b6b" if gcs > 0.33    else "#2d3436"
    lactate_color = "#ff6b6b" if lactate > 0.50 else "#2d3436"

    children = [
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
            ], style={"marginRight": "10px"}),
            html.Div([
                html.Small("O₂ SAT", style={"fontSize": "0.55rem", "color": "#b2bec3", "fontWeight": "700"}),
                html.Div(f"{1.0-spo2:.2f}", style={"fontSize": "0.8rem", "fontWeight": "600"}),
            ], style={"marginRight": "10px"}),
            html.Div([
                html.Small("HR DEV", style={"fontSize": "0.55rem", "color": "#b2bec3", "fontWeight": "700"}),
                html.Div(f"{hr:.2f}", style={"fontSize": "0.8rem", "fontWeight": "600"}),
            ], style={"marginRight": "10px"}),
            html.Div([
                html.Small("GCS ▼", style={"fontSize": "0.55rem", "color": "#b2bec3", "fontWeight": "700"}),
                html.Div(f"{gcs:.2f}", style={"fontSize": "0.8rem", "fontWeight": "600", "color": gcs_color}),
            ], style={"marginRight": "10px"}),
            html.Div([
                html.Small("LACT", style={"fontSize": "0.55rem", "color": "#b2bec3", "fontWeight": "700"}),
                html.Div(f"{lactate:.2f}", style={"fontSize": "0.8rem", "fontWeight": "600", "color": lactate_color}),
            ]),
        ], className="d-flex"),
    ]
    if reason:
        reason_bits = [
            html.Small(reason, style={"color": "#636e72", "fontSize": "0.62rem",
                                      "fontStyle": "italic", "marginTop": "6px", "display": "block"}),
        ]
        if is_fallback:
            reason_bits.append(html.Span(" GREEDY PLACED", className="fallback-badge"))
        children.append(html.Div(reason_bits))

    return html.Div(children, className=f"bed-card {pulse_class}")

def resource_column(title, icon, color, capacity, assigned, df):
    used = len(assigned)
    cards = []
    for a in assigned:
        idx  = a["patient_idx"]
        pid  = df.loc[idx, "patient_id"] if idx < len(df) else f"P{idx}"
        bp      = float(df.loc[idx, "bp_deviation"])  if idx < len(df) else 0.0
        spo2    = float(df.loc[idx, "spo2_deficit"])   if idx < len(df) else 0.0
        hr      = float(df.loc[idx, "hr_deviation"])   if (idx < len(df) and "hr_deviation" in df.columns)  else 0.0
        gcs     = float(df.loc[idx, "gcs_deficit"])    if (idx < len(df) and "gcs_deficit"  in df.columns)  else 0.0
        lactate = float(df.loc[idx, "lactate"])        if (idx < len(df) and "lactate"       in df.columns)  else 0.0
        reason  = a.get("reason", "")
        is_fb   = a.get("fallback", False)
        cards.append(bed_card(pid, a["urgency"], bp, spo2, hr, gcs, lactate, reason, is_fb))
    
    free = capacity - used
    if free > 0:
        cards.append(html.Div([
            html.Span("🛏️", style={"fontSize": "1.4rem", "marginRight": "10px"}),
            html.Span(f"{free}", style={
                "fontFamily": "JetBrains Mono", "fontWeight": "700",
                "fontSize": "1.6rem", "color": "#b2bec3", "lineHeight": "1",
            }),
            html.Span(f"  free bed{'s' if free != 1 else ''}", style={
                "fontSize": "0.8rem", "color": "#b2bec3", "fontWeight": "600",
            }),
        ], className="d-flex align-items-center justify-content-center",
           style={
               "border": "2px dashed #dfe6e9", "borderRadius": "12px",
               "padding": "16px", "margin": "4px 0",
               "background": "#f8f9fa",
           }))

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


def staff_chip(staff_id, role_name, skill, fatigue):
    """A compact staff card showing fatigue level via a colour-coded left border."""
    if fatigue >= 0.7:
        fatigue_cls = "fatigue-high"
        fatigue_label = "TIRED"
    elif fatigue >= 0.4:
        fatigue_cls = "fatigue-mid"
        fatigue_label = "OK"
    else:
        fatigue_cls = "fatigue-low"
        fatigue_label = "FRESH"

    return html.Div([
        html.Div([
            html.Span(staff_id, style={"fontWeight": "700", "fontFamily": "JetBrains Mono", "fontSize": "0.8rem"}),
            html.Span(fatigue_label, style={
                "fontSize": "0.6rem", "fontWeight": "700", "marginLeft": "6px",
                "color": "#ff6b6b" if fatigue >= 0.7 else "#f1c40f" if fatigue >= 0.4 else "#1abc9c"
            }),
        ], className="d-flex align-items-center"),
        html.Small(role_name, style={"color": "#636e72", "display": "block", "fontWeight": "600"}),
        html.Small(f"Skill {skill:.2f} · Fatigue {fatigue:.2f}",
                   style={"color": "#b2bec3", "display": "block", "fontSize": "0.65rem"}),
    ], className=f"staff-chip {fatigue_cls}")


# ── Section header helper ────────────────────────────────────────────────────

def section_header(num, icon, title, description):
    """Numbered section heading with a plain-English subtitle."""
    return html.Div([
        html.Div([
            html.Span(f"{num:02d}", style={
                "fontFamily": "JetBrains Mono", "fontWeight": "700",
                "fontSize": "0.8rem", "color": "#b2bec3",
                "border": "2px solid #dfe6e9", "borderRadius": "6px",
                "padding": "1px 8px", "marginRight": "10px",
                "letterSpacing": "0.08em",
            }),
            html.Span(icon + "  ", style={"fontSize": "1.1rem"}),
            html.Span(title, style={
                "fontWeight": "700", "fontSize": "1.0rem", "color": "#2d3436",
            }),
        ], className="d-flex align-items-center mb-1"),
        html.P(description, style={
            "color": "#636e72", "fontSize": "0.82rem",
            "margin": "0 0 14px 50px", "fontWeight": "500", "lineHeight": "1.5",
        }),
    ], style={"marginBottom": "4px"})


def waitlist_card(patient_id, urgency, reason, position):
    """Compact card for a patient who is over-capacity and waiting."""
    return html.Div([
        html.Div([
            dbc.Badge(f"#{position}", color="danger",
                      style={"marginRight": "8px", "borderRadius": "6px",
                             "fontFamily": "JetBrains Mono", "fontSize": "0.7rem"}),
            html.Span(patient_id, style={"fontWeight": "700",
                                         "fontFamily": "JetBrains Mono"}),
        ], className="d-flex align-items-center mb-1"),
        html.Div(f"Urgency: {urgency:.3f}",
                 style={"fontSize": "0.8rem", "color": "#ff6b6b", "fontWeight": "700"}),
        html.Div(reason or "Over capacity",
                 style={"fontSize": "0.75rem", "color": "#636e72", "fontStyle": "italic"}),
    ], className="waitlist-card")


def staff_ward_column(title, icon, ward_name, ward_assignments, df_staff):
    """One column in the staff grid — all staff assigned to a particular ward."""
    chips = []
    if not ward_assignments:
        chips.append(html.Div("No staff assigned",
                              style={"color": "#b2bec3", "fontSize": "0.8rem", "fontWeight": "600",
                                     "padding": "20px", "textAlign": "center"}))
    else:
        for a in ward_assignments:
            chips.append(staff_chip(a["staff_id"], a["role_name"], a["skill_level"], a["fatigue_score"]))

    return dbc.Col([
        html.Div([
            html.Span(icon, style={"fontSize": "1.2rem", "marginRight": "8px"}),
            html.Span(title, style={"fontWeight": "700", "color": "white", "fontSize": "0.95rem"}),
            dbc.Badge(f"{len(ward_assignments)} assigned", color="light", text_color="dark",
                      style={"borderRadius": "8px", "fontWeight": "700", "marginLeft": "auto"}),
        ], style={
            "background": "#2d3436", "border": "3px solid #2d3436",
            "borderRadius": "14px 14px 0 0", "padding": "12px 18px",
            "display": "flex", "alignItems": "center",
        }),
        html.Div(chips, style={
            "background": "white", "border": "3px solid #2d3436", "borderTop": "none",
            "borderRadius": "0 0 14px 14px", "padding": "14px",
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

@keyframes qs-spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

.qs-spinner-wrap {
    padding: 40px 48px;
    background: white;
    border-radius: 20px;
    border: 4px solid #2d3436;
    box-shadow: 10px 10px 0 #2d3436;
    text-align: center;
}

.qs-spinner-icon {
    font-size: 2.8rem;
    display: inline-block;
    animation: qs-spin 1.8s linear infinite;
    line-height: 1;
    margin-bottom: 12px;
}

.log-panel-box {
    background: #0d1117;
    color: #7ee787;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    border: 3px solid #2d3436;
    border-radius: 14px;
    padding: 16px 20px;
    box-shadow: 6px 6px 0px #2d3436;
    white-space: pre-wrap;
    word-break: break-all;
    max-height: 420px;
    overflow-y: auto;
    line-height: 1.75;
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

    # ── Global loading overlay (visible while run_and_store executes) ────
    dcc.Loading(
        fullscreen=True,
        overlay_style={
            "visibility": "visible",
            "backgroundColor": "rgba(253,250,245,0.92)",
            "zIndex": 9999,
        },
        delay_show=100,
        custom_spinner=html.Div([
            html.Div("⚛️", className="qs-spinner-icon"),
            html.H4("Running Quantum Pipeline…",
                    style={"fontFamily": "Fredoka", "fontWeight": "700",
                           "color": "#2d3436", "margin": "0 0 6px 0"}),
            html.P("QSVM kernel · QUBO optimizer · ML-KEM-768 encryption",
                   style={"color": "#636e72", "fontSize": "0.85rem", "margin": 0}),
        ], className="qs-spinner-wrap"),
        children=html.Div(id="loading-trigger", style={"display": "none"}),
    ),

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

        # ── STEP 0 — Configure & Run ─────────────────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Span("STEP 0", style={
                                "fontFamily": "JetBrains Mono", "fontWeight": "700",
                                "fontSize": "0.75rem", "color": "#b2bec3",
                                "letterSpacing": "0.1em", "marginRight": "10px",
                                "border": "2px solid #dfe6e9", "borderRadius": "6px",
                                "padding": "1px 8px",
                            }),
                            html.Span("Configure & Run",
                                      style={"fontWeight": "700", "fontSize": "1rem",
                                             "color": "#2d3436"}),
                        ], className="d-flex align-items-center mb-1"),
                        html.P(
                            "Set the air-quality index (AQI) to simulate a pollution surge, choose how "
                            "many patients arrive, then press Optimize to run all 8 pipeline stages.",
                            style={"color": "#636e72", "fontSize": "0.82rem",
                                   "margin": "0 0 16px 0", "lineHeight": "1.5"},
                        ),
                        dbc.Row([
                            dbc.Col([
                                html.Label("AQI / Smog Intensity (PM2.5)",
                                           style={"color": "#2d3436", "fontSize": "0.85rem",
                                                  "fontWeight": "700"}),
                                html.Small(
                                    "Higher AQI \u2192 more respiratory cases \u2192 higher urgency scores",
                                    style={"color": "#636e72", "display": "block",
                                           "marginBottom": "4px", "fontSize": "0.75rem"},
                                ),
                                dcc.Slider(id="aqi-slider", min=0, max=500, step=10, value=50,
                                           marks={0: "Clear", 250: "Moderate", 500: "Surge"},
                                           tooltip={"placement": "bottom", "always_visible": False},
                                           className="mt-1"),
                            ], width=5),
                            dbc.Col([
                                html.Label("Patient Intake Count",
                                           style={"color": "#2d3436", "fontSize": "0.85rem",
                                                  "fontWeight": "700"}),
                                html.Small(
                                    "\u26a0\ufe0f  More patients = longer QSVM kernel computation",
                                    style={"color": "#f39c12", "display": "block",
                                           "marginBottom": "4px", "fontSize": "0.75rem",
                                           "fontWeight": "600"},
                                ),
                                html.Div(
                                    dcc.Slider(id="patients-slider", min=10, max=300, step=10, value=100,
                                               marks={10: "10", 100: "100", 200: "200", 300: "300"},
                                               className="mt-1"),
                                    style={"paddingRight": "20px"},
                                ),
                            ], width=4),
                            dbc.Col([
                                dbc.Button("Optimize Now \ud83d\ude80", id="run-btn",
                                           className="run-button w-100 py-3",
                                           style={"marginTop": "1.8rem", "fontSize": "0.9rem"}),
                            ], width=3),
                        ])
                    ])
                ], className="command-card mb-4")
            ], width=12),
        ]),

        # ── KPI snapshot ──────────────────────────────────────────────────
        dbc.Row(id="metric-cards", className="mb-3"),

        # ── Tabbed sections ──────────────────────────────────────────────
        dbc.Tabs(
            id="main-tabs",
            active_tab="tab-beds",
            className="mb-0",
            style={"borderBottom": "2px solid #dfe6e9", "marginTop": "8px"},
            children=[

                # ── Tab 01 — Beds ─────────────────────────────────────
                dbc.Tab(
                    label="\U0001f3e8  Bed Allocation",
                    tab_id="tab-beds",
                    label_style={"fontWeight": "700", "fontSize": "0.85rem"},
                    children=html.Div([
                        section_header(
                            1, "\U0001f3e8", "Quantum Bed Allocation",
                            "The QUBO optimizer scores every patient via the QSVM urgency model, "
                            "then assigns each to the most appropriate ward \u2014 ICU / Trauma, "
                            "Ventilator Unit, or General Ward.",
                        ),
                        dcc.Loading(
                            id="loading-beds",
                            type="cube",
                            color="#ff6b6b",
                            children=html.Div(id="bed-grid"),
                        ),
                    ], style={"paddingTop": "20px"}),
                ),

                # ── Tab 02 — Patients ─────────────────────────────────
                dbc.Tab(
                    label="\U0001f4cb  Patient Queue",
                    tab_id="tab-patients",
                    label_style={"fontWeight": "700", "fontSize": "0.85rem"},
                    children=html.Div([
                        section_header(
                            2, "\U0001f4cb", "Patient Triage Queue",
                            "All patients ranked by the QSVM urgency score (0\u20131). "
                            "Red rows are critical (score \u2265 0.7) and are prioritised by the optimizer.",
                        ),
                        html.Div(
                            "Column guide \u2014  "
                            "BP \u0394: blood-pressure deviation  \u00b7  "
                            "O\u2082 SAT: oxygen saturation  \u00b7  "
                            "HR DEV: heart-rate deviation  \u00b7  "
                            "RESP: respiratory rate  \u00b7  "
                            "GCS \u25bc: Glasgow Coma Score deficit (higher = worse)  \u00b7  "
                            "LACTATE: serum lactate (higher = worse)  \u00b7  "
                            "SCORE: QSVM urgency 0\u20131",
                            style={
                                "background": "#f8f9fa", "border": "2px solid #dfe6e9",
                                "borderRadius": "10px", "padding": "8px 14px",
                                "fontSize": "0.76rem", "color": "#636e72",
                                "fontFamily": "JetBrains Mono", "marginBottom": "12px",
                                "lineHeight": "1.8",
                            },
                        ),
                        html.Div(id="patient-table"),
                    ], style={"paddingTop": "20px"}),
                ),

                # ── Tab 03 — Staff ────────────────────────────────────
                dbc.Tab(
                    label="\U0001f468\u200d\u2695\ufe0f  Staff Deployment",
                    tab_id="tab-staff",
                    label_style={"fontWeight": "700", "fontSize": "0.85rem"},
                    children=html.Div([
                        section_header(
                            3, "\U0001f468\u200d\u2695\ufe0f", "Staff Deployment \u2014 QUBO Stage 2",
                            "A second QUBO problem matches qualified clinical staff to each ward, "
                            "balancing role requirements, skill level, and real-time fatigue scores.",
                        ),
                        dcc.Loading(
                            id="loading-staff",
                            type="cube",
                            color="#1abc9c",
                            children=html.Div(id="staff-grid"),
                        ),
                    ], style={"paddingTop": "20px"}),
                ),

                # ── Tab 04 — Security ─────────────────────────────────
                dbc.Tab(
                    label="\U0001f6e1\ufe0f  Security",
                    tab_id="tab-security",
                    label_style={"fontWeight": "700", "fontSize": "0.85rem"},
                    children=html.Div([
                        section_header(
                            4, "\U0001f6e1\ufe0f", "Post-Quantum Encryption (NIST FIPS 203)",
                            "Every patient record is encrypted on-device with ML-KEM-768 key exchange "
                            "and AES-256-GCM symmetric encryption \u2014 resistant to future quantum attacks.",
                        ),
                        html.Div(id="security-panel"),
                    ], style={"paddingTop": "20px"}),
                ),

                # ── Tab 05 — Log ──────────────────────────────────────
                dbc.Tab(
                    label="\U0001f4df  Pipeline Log",
                    tab_id="tab-log",
                    label_style={"fontWeight": "700", "fontSize": "0.85rem"},
                    children=html.Div([
                        section_header(
                            5, "\U0001f4df", "Pipeline Log",
                            "Raw verbose output of all 8 stages from the last optimization run. "
                            "Use this to check timings and verify each algorithm step completed correctly.",
                        ),
                        html.Div(id="log-panel"),
                    ], style={"paddingTop": "20px"}),
                ),
            ],
        ),

        html.Div(style={"height": "60px"}),

    ], fluid=True),

], style={"background": "#fdfaf5", "minHeight": "100vh"})


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("pipeline-store", "data"),
    Output("loading-trigger", "children"),
    Input("run-btn", "n_clicks"),
    State("aqi-slider", "value"),
    State("patients-slider", "value"),
    prevent_initial_call=True,
)
def run_and_store(_, aqi, n_patients):
    t0 = time.time()
    log_buf = io.StringIO()
    _old_stdout = sys.stdout
    sys.stdout = log_buf
    try:
        results = run_pipeline(aqi_level=float(aqi), n_patients=int(n_patients), verbose=True)
    finally:
        sys.stdout = _old_stdout
    elapsed = round(time.time() - t0, 1)
    log_text = log_buf.getvalue()
    staff_df = results["staff_df"]
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
        # Stage 2 staff
        "staff_df":         staff_df.to_json(orient="split"),
        "staff_allocation": results["staff_allocation"],
        "staff_metrics":    results["staff_metrics"],
        "stage1_solve_ms":  results["stage1_solve_ms"],
        "stage2_solve_ms":  results["stage2_solve_ms"],
        "waitlist":         results.get("waitlist", []),
        "log_text":         log_text,
        "elapsed":          elapsed,
    }, ""


@app.callback(
    Output("metric-cards", "children"),
    Output("bed-grid", "children"),
    Output("patient-table", "children"),
    Output("security-panel", "children"),
    Output("staff-grid", "children"),
    Output("last-update", "children"),
    Output("log-panel", "children"),
    Input("pipeline-store", "data"),
)
def update_ui(data):
    if data is None:
        # Show pipeline overview while waiting
        overview = dbc.Card(dbc.CardBody([
            html.H5("How the pipeline works",
                    style={"fontWeight": "700", "color": "#2d3436", "marginBottom": "16px"}),
            *[
                html.Div([
                    html.Span(f"{n:02d}", style={
                        "fontFamily": "JetBrains Mono", "fontWeight": "700",
                        "fontSize": "0.8rem", "color": "#b2bec3",
                        "border": "2px solid #dfe6e9", "borderRadius": "6px",
                        "padding": "1px 7px", "marginRight": "10px",
                    }),
                    html.Span(icon + "  ", style={"fontSize": "1.1rem"}),
                    html.Span(label, style={"fontWeight": "700", "color": "#2d3436",
                                            "marginRight": "8px"}),
                    html.Span(desc, style={"color": "#636e72", "fontSize": "0.82rem"}),
                ], className="d-flex align-items-center mb-3")
                for n, icon, label, desc in [
                    (1, "⚛️",  "QSVM Urgency Scoring",
                     "Quantum kernel ranks every patient by clinical criticality"),
                    (2, "🏥",  "QUBO Bed Allocation",
                     "Quantum annealer assigns patients to ICU / Vent / General Ward"),
                    (3, "👨\u200d⚕️", "QUBO Staff Deployment",
                     "Second QUBO matches staff to wards by skill and fatigue"),
                    (4, "🛡️",  "ML-KEM-768 Encryption",
                     "Every record encrypted with NIST FIPS 203 post-quantum crypto"),
                    (5, "#️⃣",  "Audit Hash",
                     "SHA-512 proof of allocation integrity stored in the ledger"),
                ]
            ],
            html.P("Set your parameters above and press Optimize Now to begin.",
                   style={"color": "#636e72", "fontSize": "0.85rem",
                          "margin": "8px 0 0 0", "fontStyle": "italic"}),
        ]), className="command-card")
        return [], overview, [], [], [], "", None

    df       = pd.read_json(io.StringIO(data["df"]), orient="split")
    q_alloc  = data["q_alloc"]
    sec      = data["security"]
    aqi      = data["aqi"]
    n        = data["n_patients"]
    sm       = data.get("staff_metrics", {})
    s_alloc  = data.get("staff_allocation", [])
    waitlist = data.get("waitlist", [])

    # ── Metric cards ─────────────────────────────────────────────────────
    urgencies = [a["urgency"] for a in q_alloc]
    n_critical = sum(1 for u in urgencies if u >= 0.7)
    sam_score  = sm.get("skill_acuity_match", 0)

    cards = [
        metric_card("Live Queue", f"{n:02d}", "Patients being triaged", "#2d3436", "👤"),
        metric_card("Crisis Alert", f"{n_critical}", "Critical patients found", "#ff6b6b", "🚨"),
        metric_card("Skill Match", f"{sam_score:.3f}", "Skill-acuity score (Stage 2)", "#1abc9c", "🧑‍⚕️"),
        metric_card("Smog Level", f"{aqi}", "Local PM2.5 surge weight", "#f1c40f", "🌫️"),
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

    bed_grid = html.Div([
        dbc.Row(cols, className="mb-3"),
        # Waitlist panel (only shown when there are truly over-capacity patients)
        html.Div([
            html.Div([
                html.Span("⏳", style={"fontSize": "1.2rem", "marginRight": "8px"}),
                html.Span("PATIENT WAITLIST", style={"fontWeight": "700", "color": "#ff6b6b",
                                                      "fontSize": "0.9rem", "textTransform": "uppercase"}),
                dbc.Badge(f"{len(waitlist)} waiting", color="danger",
                          style={"marginLeft": "10px", "borderRadius": "8px", "fontWeight": "700"}),
            ], className="d-flex align-items-center mb-3"),
            dbc.Row([
                dbc.Col(waitlist_card(w["patient_id"], w["urgency"], w["reason"], i + 1), width=4)
                for i, w in enumerate(waitlist)
            ]),
        ], style={"display": "block" if waitlist else "none"}),
    ])

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
            html.Td(f"{row.get('hr_deviation', 0):.2f}"),
            html.Td(f"{row.get('resp_rate', 0):.2f}"),
            html.Td(f"{row.get('gcs_deficit', 0):.2f}",
                    style={"color": "#ff6b6b" if row.get('gcs_deficit', 0) > 0.33 else "#2d3436", "fontWeight": "600"}),
            html.Td(f"{row.get('lactate', 0):.2f}",
                    style={"color": "#ff6b6b" if row.get('lactate', 0) > 0.50 else "#2d3436", "fontWeight": "600"}),
            html.Td(f"{u:.3f}", style={"color": "#ff6b6b" if is_critical else "#2d3436", "fontWeight": "700"}),
            html.Td(dbc.Badge("URGENT" if is_critical else "STABLE", 
                              color="danger" if is_critical else "success",
                              style={"borderRadius": "8px"})),
            html.Td(q_res, style={"fontWeight": "700", "color": "#1abc9c"}),
        ]))

    table = dbc.Table(
        [html.Thead(html.Tr([
            html.Th("PATIENT"), html.Th("BP Δ"), html.Th("O₂ SAT"),
            html.Th("HR DEV"), html.Th("RESP"), html.Th("GCS ▼"), html.Th("LACTATE"),
            html.Th("SCORE"), html.Th("STATUS"),
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

    # ── Staff grid ───────────────────────────────────────────────────────────────
    ward_map = [
        ("ICU / Trauma",    "🏥", "🤚"),
        ("Ventilator Unit",  "🪁", "🚨"),
        ("General Ward",     "🛏️", "👨‍⚕️"),
    ]
    staff_cols = []
    for ward_name, icon, _ in ward_map:
        ward_staff = [a for a in s_alloc if a.get("ward") == ward_name]
        staff_cols.append(staff_ward_column(ward_name, icon, ward_name, ward_staff, None))

    # Staff utilization metrics strip
    util_pct  = sm.get("utilization_pct", {})
    s1_ms     = data.get("stage1_solve_ms", "?")
    s2_ms     = data.get("stage2_solve_ms", "?")
    unassign  = sm.get("unassigned_count", "?")
    xqual     = sm.get("cross_qual_rate", "?")

    util_badges = [
        dbc.Badge(f"{role}: {pct}%", color="light", text_color="dark",
                  style={"marginRight": "6px", "border": "2px solid #2d3436",
                         "borderRadius": "8px", "fontWeight": "700"})
        for role, pct in util_pct.items()
    ]

    staff_metrics_strip = dbc.Card(dbc.CardBody([
        html.Div([
            html.Span("📊 Staff Utilization Metrics",
                      style={"fontWeight": "700", "fontSize": "0.85rem", "marginRight": "20px", "color": "#636e72"}),
            *util_badges,
            dbc.Badge(f"Unassigned: {unassign} pts", color="danger",
                      style={"marginRight": "6px", "borderRadius": "8px"}),
            dbc.Badge(f"Cross-qual: {xqual}%", color="warning" if float(xqual or 0) > 0 else "success",
                      style={"marginRight": "6px", "borderRadius": "8px"}),
            dbc.Badge(f"S1: {s1_ms}ms → S2: {s2_ms}ms", color="dark",
                      style={"borderRadius": "8px"}),
        ], className="d-flex flex-wrap align-items-center gap-1"),
    ]), className="command-card mb-3")

    staff_grid = html.Div([
        staff_metrics_strip,
        dbc.Row(staff_cols, className="mb-4"),
    ])

    # ── Log panel ─────────────────────────────────────────────────────────
    log_text = data.get("log_text", "")
    elapsed  = data.get("elapsed", 0)
    log_panel = dbc.Card(dbc.CardBody([
        html.Div([
            html.Span("✅ PIPELINE COMPLETE",
                      style={"fontWeight": "700", "color": "#1abc9c", "fontSize": "0.85rem"}),
            dbc.Badge(f"⏱ {elapsed}s total", color="dark",
                      style={"marginLeft": "12px", "borderRadius": "8px",
                             "fontFamily": "JetBrains Mono", "fontSize": "0.78rem"}),
        ], className="mb-2"),
        html.Pre(log_text or "(no log output)", className="log-panel-box mb-0"),
    ]), className="command-card") if log_text else None

    return cards, bed_grid, table, sec_panel, staff_grid, f"LIVE UPDATED: {ts}", log_panel


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, port=8050, host="0.0.0.0")
