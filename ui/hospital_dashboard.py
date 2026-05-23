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

.unallocated-card {
    background: #fef3c7;
    border: 3px solid #f59e0b;
    border-left: 8px solid #f59e0b;
    border-radius: 14px;
    padding: 12px 16px;
    margin-bottom: 10px;
    box-shadow: 4px 4px 0px #2d3436;
}

.fallback-badge {
    font-size: 0.6rem;
    background: #6c5ce7;
    color: #ffffff;
    border: 2px solid #5a4bd1;
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

def bed_card(patient_id, urgency, bp, spo2, hr=0.0, gcs=0.0, lactate=0.0, reason="", is_fallback=False, is_new=False):
    is_critical = urgency >= 0.7
    pulse_class = "pulse-critical" if is_critical else ""
    new_class   = "bed-card-new"   if is_new       else ""

    gcs_color     = "#ff6b6b" if gcs > 0.33    else "#2d3436"
    lactate_color = "#ff6b6b" if lactate > 0.50 else "#2d3436"

    children = [
        html.Div([
            html.Div([
                html.Small("PATIENT ID", style={"fontSize": "0.6rem", "color": "#636e72", "display": "block", "fontWeight": "700"}),
                html.Div([
                    html.Span(patient_id, style={"fontWeight": "700", "fontSize": "1.1rem"}),
                    html.Span("✨ NEW", className="new-badge") if is_new else None,
                ], className="d-flex align-items-center"),
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
            reason_bits.append(html.Span(" ⚛ QSVM-RANKED", className="fallback-badge"))
        children.append(html.Div(reason_bits))

    return html.Div(children, className=f"bed-card {pulse_class} {new_class}")

def resource_column(title, icon, color, capacity, assigned, df, latest_patient_id=None):
    # Hard guard: never show more patients than capacity
    used = min(len(assigned), capacity)
    assigned = assigned[:used]
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
        is_new  = (latest_patient_id is not None and pid == latest_patient_id)
        cards.append(bed_card(pid, a["urgency"], bp, spo2, hr, gcs, lactate, reason, is_fb, is_new))

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

    # Badge colour: red = full, yellow = one bed left, green = available
    if used >= capacity:
        badge_color, badge_text_color = "danger", "white"
        badge_label = f"{used}/{capacity} FULL"
    elif used >= capacity - 1:
        badge_color, badge_text_color = "warning", "dark"
        badge_label = f"{used}/{capacity} — 1 bed left"
    else:
        badge_color, badge_text_color = "success", "white"
        badge_label = f"{used}/{capacity}"

    return dbc.Col([
        html.Div([
            html.Div([
                html.Span(icon, style={"fontSize": "1.3rem", "marginRight": "10px"}),
                html.Span(title, style={"fontWeight": "700", "fontSize": "1rem", "color": "white"}),
            ]),
            dbc.Badge(badge_label, color=badge_color, text_color=badge_text_color,
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


def staff_chip(staff_id, role_name, skill, fatigue, patient_ids=None):
    """A compact staff card showing fatigue level and assigned patients."""
    if fatigue >= 0.7:
        fatigue_cls = "fatigue-high"
        fatigue_label = "TIRED"
    elif fatigue >= 0.4:
        fatigue_cls = "fatigue-mid"
        fatigue_label = "OK"
    else:
        fatigue_cls = "fatigue-low"
        fatigue_label = "FRESH"

    patient_ids = patient_ids or []
    pts_badges = [
        html.Span(pid, style={
            "fontSize": "0.6rem", "fontWeight": "700",
            "background": "#f1f2f6", "borderRadius": "5px",
            "padding": "1px 5px", "marginRight": "3px", "marginTop": "2px",
            "fontFamily": "JetBrains Mono", "display": "inline-block",
        })
        for pid in patient_ids
    ]

    return html.Div([
        html.Div([
            html.Span(staff_id, style={"fontWeight": "700", "fontFamily": "JetBrains Mono", "fontSize": "0.8rem"}),
            html.Span(fatigue_label, style={
                "fontSize": "0.6rem", "fontWeight": "700", "marginLeft": "6px",
                "color": "#ff6b6b" if fatigue >= 0.7 else "#f1c40f" if fatigue >= 0.4 else "#1abc9c"
            }),
            html.Span(f"{len(patient_ids)} pt{'s' if len(patient_ids) != 1 else ''}", style={
                "fontSize": "0.6rem", "color": "#636e72", "marginLeft": "auto",
                "fontWeight": "600",
            }),
        ], className="d-flex align-items-center"),
        html.Small(role_name, style={"color": "#636e72", "display": "block", "fontWeight": "600"}),
        html.Small(f"Skill {skill:.2f} · Fatigue {fatigue:.2f}",
                   style={"color": "#b2bec3", "display": "block", "fontSize": "0.65rem"}),
        html.Div(pts_badges, style={"marginTop": "4px", "display": "flex", "flexWrap": "wrap"}),
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


def unallocated_card(patient_id, urgency, position):
    """Compact card for a patient who could not be allocated due to capacity limits."""
    return html.Div([
        html.Div([
            dbc.Badge(f"#{position}", color="warning",
                      style={"marginRight": "8px", "borderRadius": "6px",
                             "fontFamily": "JetBrains Mono", "fontSize": "0.7rem"}),
            html.Span(patient_id, style={"fontWeight": "700",
                                         "fontFamily": "JetBrains Mono"}),
        ], className="d-flex align-items-center mb-1"),
        html.Div(f"Urgency: {urgency:.3f}",
                 style={"fontSize": "0.8rem", "color": "#f59e0b", "fontWeight": "700"}),
        html.Div("⏳ No bed available — insufficient capacity",
                 style={"fontSize": "0.75rem", "color": "#636e72", "fontStyle": "italic"}),
    ], className="unallocated-card")


def staff_ward_column(title, icon, ward_name, ward_assignments, df_staff):
    """One column in the staff grid — unique staff grouped with their patient list."""
    chips = []
    if not ward_assignments:
        chips.append(html.Div("No staff assigned",
                              style={"color": "#b2bec3", "fontSize": "0.8rem", "fontWeight": "600",
                                     "padding": "20px", "textAlign": "center"}))
    else:
        # Group assignments by staff member so each person appears once
        seen = {}  # staff_id → dict with aggregated patient list
        for a in ward_assignments:
            sid = a["staff_id"]
            if sid not in seen:
                seen[sid] = {**a, "patient_ids": []}
            seen[sid]["patient_ids"].append(a["patient_id"])

        for entry in seen.values():
            chips.append(staff_chip(
                entry["staff_id"], entry["role_name"],
                entry["skill_level"], entry["fatigue_score"],
                entry["patient_ids"],
            ))

    n_staff_shown = len(seen) if ward_assignments else 0
    n_pts_covered = len(ward_assignments)

    return dbc.Col([
        html.Div([
            html.Span(icon, style={"fontSize": "1.2rem", "marginRight": "8px"}),
            html.Span(title, style={"fontWeight": "700", "color": "white", "fontSize": "0.95rem"}),
            html.Div([
                dbc.Badge(f"{n_staff_shown} staff", color="light", text_color="dark",
                          style={"borderRadius": "8px", "fontWeight": "700", "marginRight": "4px"}),
                dbc.Badge(f"{n_pts_covered} pts", color="info",
                          style={"borderRadius": "8px", "fontWeight": "700"}),
            ], style={"marginLeft": "auto", "display": "flex"}),
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

/* ── Manual Patient Addition ── */
.bed-card-new {
    border: 3px solid #9b59b6 !important;
    background: linear-gradient(135deg, #f5eeff 0%, #ffffff 100%) !important;
    box-shadow: 4px 4px 0px #9b59b6 !important;
    animation: new-glow 1.8s ease-in-out infinite;
}

@keyframes new-glow {
    0%, 100% { box-shadow: 4px 4px 0px #9b59b6, 0 0 0px 0px rgba(155,89,182,0); }
    50%       { box-shadow: 4px 4px 0px #9b59b6, 0 0 20px 6px rgba(155,89,182,0.35); }
}

.new-badge {
    background: #9b59b6;
    color: white;
    border: 2px solid #2d3436;
    border-radius: 6px;
    padding: 1px 7px;
    font-size: 0.58rem;
    font-weight: 700;
    margin-left: 6px;
    letter-spacing: 0.05em;
    vertical-align: middle;
}

.add-btn {
    background: #9b59b6 !important;
    border: 3px solid #2d3436 !important;
    border-radius: 14px !important;
    box-shadow: 6px 6px 0px #2d3436 !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    text-transform: uppercase;
    transition: all 0.1s ease;
}

.add-btn:active {
    transform: translate(2px, 2px);
    box-shadow: 2px 2px 0px #2d3436 !important;
}

.allocation-result-banner {
    background: linear-gradient(135deg, #f5eeff 0%, #fff 100%);
    border: 4px solid #9b59b6;
    border-radius: 18px;
    box-shadow: 8px 8px 0 #2d3436;
    padding: 24px 32px;
    margin-top: 20px;
}

tr.patient-row-new td {
    background: #f5eeff !important;
    border-top: 2px solid #9b59b6 !important;
    border-bottom: 2px solid #9b59b6 !important;
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
    title="🏥  Q-Hospital",
)
app.index_string = app.index_string.replace(
    "</head>",
    f"<style>{CUSTOM_STYLE}</style></head>",
)

app.layout = html.Div([
    dcc.Store(id="pipeline-store"),
    dcc.Store(id="manual-counter", data=0),

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
                        html.H3("🏥  Q-Hospital",
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
                            "TEMP \u0394: fever / hypothermia delta (>0.60 = flagged)  \u00b7  "
                            "AQI: local PM2.5 reading (>0.50 = surge)  \u00b7  "
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

                # ── Tab 06 — Add Patient ──────────────────────────────
                dbc.Tab(
                    label="➕  Add Patient",
                    tab_id="tab-add-patient",
                    label_style={"fontWeight": "700", "fontSize": "0.85rem",
                                 "color": "#9b59b6"},
                    children=html.Div([
                        section_header(
                            6, "🧑‍⚕️", "Manual Patient Entry",
                            "Enter a patient's vitals manually. The quantum pipeline will "
                            "score their urgency and allocate them to the optimal ward. "
                            "Make sure to run Optimize Now first so a baseline exists.",
                        ),
                        dbc.Card(dbc.CardBody([
                            dbc.Row([
                                # ── Left column: haemodynamic ──────────
                                dbc.Col([
                                    html.H6("❤️  Haemodynamic & Respiratory",
                                            style={"fontWeight": "700", "color": "#2d3436",
                                                   "marginBottom": "16px"}),
                                    html.Label("BP Deviation  (0 = normal · 100 = max deviation)",
                                               style={"fontSize": "0.8rem", "fontWeight": "700",
                                                      "color": "#636e72"}),
                                    dcc.Slider(id="inp-bp", min=0, max=100, step=1, value=20,
                                               marks={0: "0", 50: "50", 100: "100"},
                                               tooltip={"placement": "bottom", "always_visible": True},
                                               className="mb-3"),
                                    html.Label("SpO₂ Deficit  (0 = normal · 100 = severe hypoxia)",
                                               style={"fontSize": "0.8rem", "fontWeight": "700",
                                                      "color": "#636e72"}),
                                    dcc.Slider(id="inp-spo2", min=0, max=100, step=1, value=10,
                                               marks={0: "0", 50: "50", 100: "100"},
                                               tooltip={"placement": "bottom", "always_visible": True},
                                               className="mb-3"),
                                    html.Label("Temperature Δ  (0 = afebrile · 100 = extreme)",
                                               style={"fontSize": "0.8rem", "fontWeight": "700",
                                                      "color": "#636e72"}),
                                    dcc.Slider(id="inp-temp", min=0, max=100, step=1, value=15,
                                               marks={0: "0", 50: "50", 100: "100"},
                                               tooltip={"placement": "bottom", "always_visible": True},
                                               className="mb-3"),
                                    html.Label("AQI Exposure  (0 = clean air · 100 = hazardous PM2.5)",
                                               style={"fontSize": "0.8rem", "fontWeight": "700",
                                                      "color": "#636e72"}),
                                    dcc.Slider(id="inp-aqi", min=0, max=100, step=1, value=10,
                                               marks={0: "0", 50: "50", 100: "100"},
                                               tooltip={"placement": "bottom", "always_visible": True},
                                               className="mb-3"),
                                ], width=6),
                                # ── Right column: neurological / shock ─
                                dbc.Col([
                                    html.H6("🧠  Neurological & Shock Markers",
                                            style={"fontWeight": "700", "color": "#2d3436",
                                                   "marginBottom": "16px"}),
                                    html.Label("HR Deviation  (0 = normal · 100 = extreme tachycardia)",
                                               style={"fontSize": "0.8rem", "fontWeight": "700",
                                                      "color": "#636e72"}),
                                    dcc.Slider(id="inp-hr", min=0, max=100, step=1, value=20,
                                               marks={0: "0", 50: "50", 100: "100"},
                                               tooltip={"placement": "bottom", "always_visible": True},
                                               className="mb-3"),
                                    html.Label("Resp Rate  (0 = normal · 100 = severe tachypnoea)",
                                               style={"fontSize": "0.8rem", "fontWeight": "700",
                                                      "color": "#636e72"}),
                                    dcc.Slider(id="inp-resp", min=0, max=100, step=1, value=20,
                                               marks={0: "0", 50: "50", 100: "100"},
                                               tooltip={"placement": "bottom", "always_visible": True},
                                               className="mb-3"),
                                    html.Label("GCS Deficit  (0 = alert · 100 = deep coma)",
                                               style={"fontSize": "0.8rem", "fontWeight": "700",
                                                      "color": "#636e72"}),
                                    dcc.Slider(id="inp-gcs", min=0, max=100, step=1, value=5,
                                               marks={0: "0", 33: "coma", 100: "100"},
                                               tooltip={"placement": "bottom", "always_visible": True},
                                               className="mb-3"),
                                    html.Label("Lactate  (0 = normal · 100 = severe shock / hypoperfusion)",
                                               style={"fontSize": "0.8rem", "fontWeight": "700",
                                                      "color": "#636e72"}),
                                    dcc.Slider(id="inp-lactate", min=0, max=100, step=1, value=10,
                                               marks={0: "0", 50: "shock", 100: "100"},
                                               tooltip={"placement": "bottom", "always_visible": True},
                                               className="mb-3"),
                                ], width=6),
                            ]),
                            html.Hr(),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Patient Name / ID  (optional — auto-assigned if blank)",
                                               style={"fontSize": "0.8rem", "fontWeight": "700",
                                                      "color": "#636e72"}),
                                    dbc.Input(id="inp-patient-name", type="text",
                                              placeholder="e.g. John Doe or leave blank",
                                              style={"border": "3px solid #2d3436",
                                                     "borderRadius": "10px",
                                                     "fontFamily": "JetBrains Mono",
                                                     "fontSize": "0.85rem",
                                                     "fontWeight": "600"}),
                                ], width=6),
                                dbc.Col([
                                    dbc.Button("🚑  Add Patient to Queue",
                                               id="add-patient-btn",
                                               className="add-btn w-100 py-3",
                                               style={"marginTop": "1.6rem"}),
                                ], width=6),
                            ]),
                        ]), className="command-card mb-3"),
                        # ── Allocation result banner (updated by update_ui) ──
                        html.Div(id="add-patient-result"),
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
    Output("manual-counter", "data"),
    Input("run-btn", "n_clicks"),
    Input("add-patient-btn", "n_clicks"),
    State("aqi-slider", "value"),
    State("patients-slider", "value"),
    State("inp-bp",      "value"),
    State("inp-spo2",    "value"),
    State("inp-temp",    "value"),
    State("inp-aqi",     "value"),
    State("inp-hr",      "value"),
    State("inp-resp",    "value"),
    State("inp-gcs",     "value"),
    State("inp-lactate", "value"),
    State("inp-patient-name", "value"),
    State("manual-counter", "data"),
    State("pipeline-store", "data"),
    prevent_initial_call=True,
)
def run_and_store(
    _run, _add,
    aqi, n_patients,
    bp, spo2, temp, aqi_inp, hr, resp, gcs, lactate,
    patient_name, counter, existing_store,
):
    import pandas as _pd
    ctx = dash.callback_context
    triggered_id = ctx.triggered_id

    extra_df          = None
    latest_patient_id = None
    new_counter       = counter or 0

    if triggered_id == "add-patient-btn":
        # Build the manual patient row
        new_counter += 1
        raw_name = (patient_name or "").strip()
        pid = raw_name if raw_name else f"MANUAL-{new_counter:03d}"
        latest_patient_id = pid

        def _v(x):  # slider 0-100 → 0.0-1.0, default 0
            return float(x or 0) / 100.0

        new_row = {
            "bp_deviation":      _v(bp),
            "spo2_deficit":      _v(spo2),
            "temperature_delta": _v(temp),
            "aqi_pm25":          _v(aqi_inp),
            "hr_deviation":      _v(hr),
            "resp_rate":         _v(resp),
            "gcs_deficit":       _v(gcs),
            "lactate":           _v(lactate),
            "label":             1,
            "patient_id":        pid,
        }
        extra_df = _pd.DataFrame([new_row])

        # If we already have a run, keep existing n_patients & aqi
        if existing_store is not None:
            aqi        = existing_store.get("aqi", aqi)
            n_patients = existing_store.get("n_patients", n_patients)

    t0 = time.time()
    log_buf = io.StringIO()
    _old_stdout = sys.stdout
    sys.stdout = log_buf
    try:
        results = run_pipeline(
            aqi_level=float(aqi),
            n_patients=int(n_patients),
            verbose=True,
            extra_df=extra_df,
        )
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
        "latest_patient_id": latest_patient_id,
    }, "", new_counter


@app.callback(
    Output("metric-cards", "children"),
    Output("bed-grid", "children"),
    Output("patient-table", "children"),
    Output("security-panel", "children"),
    Output("staff-grid", "children"),
    Output("last-update", "children"),
    Output("log-panel", "children"),
    Output("add-patient-result", "children"),
    Input("pipeline-store", "data"),
)
def update_ui(data):
    if data is None:
        # Show pipeline overview while waiting
        overview = dbc.Card(dbc.CardBody([            html.H5("How the pipeline works",
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
        return [], overview, [], [], [], "", None, None

    df       = pd.read_json(io.StringIO(data["df"]), orient="split")
    q_alloc  = data["q_alloc"]
    sec      = data["security"]
    aqi      = data["aqi"]
    n        = data["n_patients"]
    sm       = data.get("staff_metrics", {})
    s_alloc  = data.get("staff_allocation", [])
    waitlist = data.get("waitlist", [])
    latest_patient_id = data.get("latest_patient_id")

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
    # Compute effective capacity per ward: min(hospital_capacity, batch_size)
    # This prevents over-allocation when batch < hospital capacity
    effective_capacity = [
        min(RESOURCE_CAPACITY[0], n),
        min(RESOURCE_CAPACITY[1], n),
        min(RESOURCE_CAPACITY[2], n),
    ]
    resource_cfg = [
        (0, "ICU / Trauma",    "🏥", "#ff6b6b", effective_capacity[0]),
        (1, "Vent Unit",       "🫁", "#f1c40f", effective_capacity[1]),
        (2, "General Ward",    "🛏️",  "#1abc9c", effective_capacity[2]),
    ]
    cols = []
    for ridx, title, icon, color, cap in resource_cfg:
        assigned = [a for a in q_alloc if a.get("resource_idx") == ridx]
        cols.append(resource_column(title, icon, color, cap, assigned, df, latest_patient_id))

    bed_grid = html.Div([
        dbc.Row(cols, className="mb-3"),
        # Unallocated patients panel (strict capacity — no beds available)
        html.Div([
            html.Div([
                html.Span("⚠️", style={"fontSize": "1.2rem", "marginRight": "8px"}),
                html.Span("UNALLOCATED PATIENTS", style={"fontWeight": "700", "color": "#f59e0b",
                                                         "fontSize": "0.9rem", "textTransform": "uppercase"}),
                dbc.Badge(f"{len(waitlist)} no capacity", color="warning",
                          style={"marginLeft": "10px", "borderRadius": "8px", "fontWeight": "700"}),
            ], className="d-flex align-items-center mb-3"),
            dbc.Row([
                dbc.Col(unallocated_card(
                    w["patient_id"], w["urgency"], i + 1,
                    displaced=w.get("displaced", False),
                    displaced_by_urgency=w.get("displaced_by_urgency"),
                    displaced_by_patient_id=w.get("displaced_by_patient_id"),
                ), width=4)
                for i, w in enumerate(waitlist)
            ]),
            html.P([
                html.Span(f"All beds are at capacity: ICU {effective_capacity[0]}/filled, Vent {effective_capacity[1]}/filled, General {effective_capacity[2]}/filled. ", style={"fontSize": "0.75rem"}),
                html.Span("These patients are on the waitlist — awaiting discharge or emergency bed availability.", style={"fontSize": "0.75rem", "fontWeight": "600", "color": "#f59e0b"}),
            ], style={"marginTop": "12px", "padding": "10px", "background": "#fef3c7", "borderRadius": "8px", "border": "1px solid #f59e0b"}),
        ], style={"display": "block" if waitlist else "none"}),
    ])

    # ── Patient intake table ──────────────────────────────────────────────
    rows = []
    for _, row in df.iterrows():
        u = float(row["urgency_score"])
        is_critical = u >= 0.7
        is_new_row  = (latest_patient_id is not None and row["patient_id"] == latest_patient_id)
        q_res = next(
            (a["resource_name"] for a in q_alloc if a["patient_idx"] == row.name),
            "⚠️ UNALLOCATED",
        )
        row_style = {"background": "#f5eeff"} if is_new_row else {}
        pid_cell = html.Td([
            html.Span(row["patient_id"], style={"fontWeight": "700", "fontFamily": "JetBrains Mono"}),
            html.Span("✨ NEW", className="new-badge") if is_new_row else None,
        ])
        rows.append(html.Tr([
            pid_cell,
            html.Td(f"{row['bp_deviation']:.2f}"),
            html.Td(f"{1.0-row['spo2_deficit']:.2f}"),
            html.Td(f"{row.get('temperature_delta', 0):.2f}",
                    style={"color": "#ff6b6b" if row.get('temperature_delta', 0) > 0.60 else "#2d3436"}),
            html.Td(f"{row.get('aqi_pm25', 0):.2f}",
                    style={"color": "#f39c12" if row.get('aqi_pm25', 0) > 0.50 else "#2d3436"}),
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
            html.Td(q_res, style={"fontWeight": "700", "color": "#1abc9c" if not is_new_row else "#9b59b6"}),
        ], style={"background": "#f5eeff", "borderTop": "2px solid #9b59b6",
                  "borderBottom": "2px solid #9b59b6"} if is_new_row else {}))

    table = dbc.Table(
        [html.Thead(html.Tr([
            html.Th("PATIENT"), html.Th("BP Δ"), html.Th("O₂ SAT"),
            html.Th("TEMP Δ"), html.Th("AQI"),
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

    # ── Add-patient result banner ─────────────────────────────────────────
    add_patient_result = None
    if latest_patient_id:
        # Find the allocation entry for this patient
        new_alloc = next(
            (a for a in q_alloc
             if df.loc[a["patient_idx"], "patient_id"] == latest_patient_id
             if a["patient_idx"] < len(df)),
            None,
        )
        if new_alloc:
            ward_name   = new_alloc.get("resource_name", "Unknown Ward")
            urgency_val = new_alloc.get("urgency", 0.0)
            reason_txt  = new_alloc.get("reason", "")
            ward_icon   = {"ICU / Trauma": "🏥", "Ventilator Unit": "🫁", "General Ward": "🛏️"}.get(ward_name, "🏨")
            is_crit     = urgency_val >= 0.7
            add_patient_result = html.Div([
                html.Div([
                    html.Span("✅ Patient Added & Allocated",
                              style={"fontWeight": "700", "fontSize": "1.1rem",
                                     "color": "#9b59b6"}),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Small("PATIENT ID", style={"fontSize": "0.7rem", "color": "#636e72",
                                                         "fontWeight": "700", "display": "block"}),
                        html.Div([
                            html.Span(latest_patient_id,
                                      style={"fontWeight": "700", "fontFamily": "JetBrains Mono",
                                             "fontSize": "1.4rem"}),
                            html.Span("✨ NEW", className="new-badge ms-2"),
                        ], className="d-flex align-items-center"),
                    ], width=4),
                    dbc.Col([
                        html.Small("ALLOCATED TO", style={"fontSize": "0.7rem", "color": "#636e72",
                                                           "fontWeight": "700", "display": "block"}),
                        html.Div([
                            html.Span(ward_icon, style={"fontSize": "1.6rem", "marginRight": "8px"}),
                            html.Span(ward_name, style={"fontWeight": "700", "fontSize": "1.2rem",
                                                         "color": "#9b59b6"}),
                        ], className="d-flex align-items-center"),
                    ], width=4),
                    dbc.Col([
                        html.Small("URGENCY SCORE", style={"fontSize": "0.7rem", "color": "#636e72",
                                                             "fontWeight": "700", "display": "block"}),
                        html.Div(f"{urgency_val:.3f}", style={
                            "fontWeight": "700", "fontSize": "1.6rem",
                            "color": "#ff6b6b" if is_crit else "#2d3436",
                            "fontFamily": "JetBrains Mono",
                        }),
                        dbc.Badge("CRITICAL — ICU PRIORITY" if is_crit else "STABLE",
                                  color="danger" if is_crit else "success",
                                  style={"borderRadius": "8px", "marginTop": "4px"}),
                    ], width=4),
                ]),
                html.Hr(style={"borderColor": "#9b59b6", "opacity": "0.3", "margin": "14px 0"}),
                html.Small(reason_txt or "Quantum allocation complete.",
                           style={"color": "#636e72", "fontStyle": "italic",
                                  "fontFamily": "JetBrains Mono", "fontSize": "0.78rem"}),
                html.Div([
                    html.Small("→ Switch to  ",
                               style={"color": "#636e72", "fontSize": "0.8rem"}),
                    html.Strong("🏢 Bed Allocation",
                                style={"color": "#9b59b6", "fontSize": "0.8rem"}),
                    html.Small("  tab to see the highlighted bed card.",
                               style={"color": "#636e72", "fontSize": "0.8rem"}),
                ], className="mt-2"),
            ], className="allocation-result-banner")

    return cards, bed_grid, table, sec_panel, staff_grid, f"LIVE UPDATED: {ts}", log_panel, add_patient_result


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, port=8050, host="0.0.0.0")
