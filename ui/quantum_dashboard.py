"""
ui/quantum_dashboard.py
Quantum Engine Visualizer — QSVM kernel, QUBO matrix, QAOA circuit, allocation diff.

Run with:
    python ui/quantum_dashboard.py
    then open  http://localhost:8051
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from pipeline import run_pipeline
from core.qsvm import NUM_QUBITS as QSVM_QUBITS

# ── Custom "Beige & Fun" Theme ────────────────────────────────────────────────

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

.nav-tabs {
    border-bottom: 4px solid var(--border-ink) !important;
    margin-bottom: 24px;
}

.nav-link {
    color: var(--text-dark) !important;
    border: none !important;
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
    padding: 12px 24px !important;
    transition: all 0.1s ease;
}

.nav-link.active {
    color: white !important;
    background: var(--border-ink) !important;
    border-radius: 12px 12px 0 0 !important;
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

.metric-value {
    font-family: 'Fredoka', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-dark) !important;
}

.section-header {
    color: var(--text-dark);
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
}

.section-header::before {
    content: '';
    display: inline-block;
    width: 6px;
    height: 24px;
    background: var(--accent-red);
    margin-right: 12px;
    border-radius: 3px;
    border: 2px solid var(--border-ink);
}

.qaoa-circuit-box {
    background: white;
    padding: 30px;
    border: 3px solid var(--border-ink);
    border-radius: 16px;
    display: flex;
    justify-content: center;
    box-shadow: inset 4px 4px 0px rgba(0,0,0,0.05);
}

.badge-quantum {
    background: var(--accent-red);
    color: white;
    border: 2px solid var(--border-ink);
    font-weight: 700;
    font-size: 0.75rem;
    padding: 5px 10px;
}

.badge-classical {
    background: var(--accent-teal);
    color: white;
    border: 2px solid var(--border-ink);
    font-weight: 700;
    font-size: 0.75rem;
    padding: 5px 10px;
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
"""

# ── App setup ─────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600;700&display=swap"
    ],
    title="⚛️ QX Engine | Quantum Dashboard",
)
app.index_string = app.index_string.replace(
    "</head>",
    f"<style>{CUSTOM_STYLE}</style></head>",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

RED    = "#ff6b6b"
TEAL   = "#1abc9c"
YELLOW = "#f1c40f"
DARK   = "#2d3436"

def info_card(title, value, subtitle="", color=RED, icon=""):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div([
            html.Span(icon, style={"fontSize": "1.3rem", "marginRight": "8px"}),
            html.Span(title, style={"color": "#636e72", "fontSize": "0.8rem", "textTransform": "uppercase", "fontWeight": "700"}),
        ], className="d-flex align-items-center mb-2"),
        html.Div(str(value), className="metric-value", style={"fontSize": "1.8rem", "lineHeight": "1", "color": color}),
        html.Div(subtitle, style={"color": "#636e72", "fontSize": "0.75rem", "marginTop": "6px", "fontWeight": "600"}),
    ]), className="command-card"), width=2)


# ── Layout ────────────────────────────────────────────────────────────────────

app.layout = html.Div([
    dcc.Store(id="q-store"),

    # ── Header ───────────────────────────────────────────────────────────
    html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("⚛️  QX Quantum Engine",
                            style={"margin": 0, "color": DARK,
                                   "fontWeight": "700"}),
                    html.Small("Algorithm Visualizer  ·  PennyLane + D-Wave",
                               style={"color": "#636e72", "fontWeight": "600"}),
                ], width=6),
                dbc.Col([
                    dbc.Badge("QSVM KERNEL",  color="danger", className="me-2", style={"borderRadius": "8px"}),
                    dbc.Badge("QUBO SOLVER",   color="dark", className="me-2", style={"borderRadius": "8px"}),
                    dbc.Badge("QAOA P=1", color="success", style={"borderRadius": "8px"}),
                ], width=6, className="d-flex align-items-center justify-content-end"),
            ], align="center"),
        ], fluid=True),
    ], className="header-blur py-3 mb-4"),

    dbc.Container([

        # ── Controls ─────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.Label("🌫️  SMOG INTENSITY",  style={"color": DARK, "fontSize": "0.85rem", "fontWeight": "700"}),
                dcc.Slider(id="q-aqi",  min=0,  max=500, step=10,  value=50,
                           marks={0: "CLEAR", 250: "MID", 500: "SURGE"},
                           tooltip={"placement": "bottom", "always_visible": False}),
            ], width=4),
            dbc.Col([
                html.Label("👤  PATIENT COUNT", style={"color": DARK, "fontSize": "0.85rem", "fontWeight": "700"}),
                html.Small(
                    "⚠️  More patients = longer QSVM kernel computation",
                    style={"color": "#f39c12", "display": "block",
                           "marginBottom": "4px", "fontSize": "0.75rem", "fontWeight": "600"},
                ),
                html.Div(
                    dcc.Slider(id="q-pts", min=10, max=300, step=10, value=100,
                               marks={10: "10", 100: "100", 200: "200", 300: "300"}),
                    style={"paddingRight": "20px"},
                ),
            ], width=4),
            dbc.Col([
                dbc.Button("EXPLODE PIPELINE ⚛️", id="q-run-btn", 
                           className="run-button w-100 py-3 mt-2"),
            ], width=4),
        ], className="mb-4"),

        # ── Loading wrapper ───────────────────────────────────────────────
        dcc.Loading(
            id="q-loading",
            type="cube",
            color=RED,
            children=html.Div([

                # Summary bar (visible once data arrives)
                html.Div(id="q-summary-bar", className="mb-4"),

                # Tabs
                dbc.Tabs(id="q-tabs", active_tab="tab-qsvm", children=[
                    dbc.Tab(label="QSVM Kernel",     tab_id="tab-qsvm"),
                    dbc.Tab(label="QUBO Matrix",     tab_id="tab-qubo"),
                    dbc.Tab(label="QAOA Circuit",    tab_id="tab-qaoa"),
                    dbc.Tab(label="Allocations",     tab_id="tab-alloc"),
                    dbc.Tab(label="Staff QUBO",      tab_id="tab-staff"),
                ]),

                html.Div(id="q-tab-content"),
            ]),
        ),

    ], fluid=True),

], style={"background": "#fdfaf5", "minHeight": "100vh"})


# ── Pipeline callback ─────────────────────────────────────────────────────────

@app.callback(
    Output("q-store", "data"),
    Input("q-run-btn", "n_clicks"),
    State("q-aqi", "value"),
    State("q-pts",  "value"),
    prevent_initial_call=True,
)
def q_run_pipeline(_, aqi, n_patients):
    r = run_pipeline(aqi_level=float(aqi), n_patients=int(n_patients), verbose=False)
    staff_df = r["staff_df"]
    return {
        "df":              r["df"].to_json(orient="split"),
        "urgency_scores":  r["urgency_scores"].tolist(),
        "q_alloc":         r["quantum_allocation"],
        "c_alloc":         r["classical_allocation"],
        "q_util":          r["quantum_utilization"],
        "c_util":          r["classical_utilization"],
        "kernel_matrix":   r["kernel_matrix"].tolist(),
        "qubo_dict":       {str(k): v for k, v in r["qubo_dict"].items()},
        "qubo_size":       len(r["urgency_scores"]) * 3,
        "alpha":           r["alpha"],
        "qsvm_f1":         r["qsvm_f1"],
        "rf_f1":           r["rf_f1"],
        "rf_urgency":      r["rf_urgency"].tolist(),
        "qaoa_circuit_b64": r["qaoa_circuit_b64"],
        "qaoa_info":       r["qaoa_info"],
        "n_patients":      n_patients,
        "aqi":             aqi,
        # Stage 2
        "staff_df":        staff_df.to_json(orient="split"),
        "staff_allocation": r["staff_allocation"],
        "staff_metrics":    r["staff_metrics"],
        "staff_qubo_dict":  {str(k): v for k, v in r["staff_qubo_dict"].items()},
        "alpha_s":          r["alpha_s"],
        "stage1_solve_ms":  r["stage1_solve_ms"],
        "stage2_solve_ms":  r["stage2_solve_ms"],
    }


# ── Summary bar + tab content ─────────────────────────────────────────────────

@app.callback(
    Output("q-summary-bar", "children"),
    Output("q-tab-content", "children"),
    Input("q-store", "data"),
    Input("q-tabs",  "active_tab"),
)
def render_tabs(data, active_tab):
    if data is None:
        overview = dbc.Card(dbc.CardBody([
            html.H5("What the quantum engine visualizes",
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
                    (1, "⚛️",  "QSVM Kernel",
                     "Heatmap of quantum kernel overlaps in Hilbert space + urgency vs classical RF"),
                    (2, "🧩", "QUBO Matrix",
                     "Ising spin-glass energy landscape for the bed-allocation optimisation problem"),
                    (3, "🌀", "QAOA Circuit",
                     "Variational quantum circuit diagram at p=1 Trotter depth"),
                    (4, "📊", "Allocations",
                     "Quantum vs classical assignment diff table and ward utilisation donuts"),
                    (5, "👨\u200d⚕️", "Staff QUBO",
                     "Stage 2 QUBO matrix and staff-to-ward assignment results"),
                ]
            ],
            html.P("Set your parameters above and press EXPLODE PIPELINE to begin.",
                   style={"color": "#636e72", "fontSize": "0.85rem",
                          "margin": "8px 0 0 0", "fontStyle": "italic"}),
        ]), className="command-card")
        return [], overview

    df     = pd.read_json(io.StringIO(data["df"]), orient="split")
    alpha  = data["alpha"]
    n      = data["n_patients"]
    n_vars = n * 3

    # ── Summary bar ───────────────────────────────────────────────────────
    summary = dbc.Row([
        info_card("QUBO Nodes", f"{n_vars}", "Spin variables", DARK, "🕸️"),
        info_card("α Penalty",  f"{alpha:.1f}", "Dynamic β-dominance", RED, "⚖️"),
        info_card("QSVM F1",    f"{data['qsvm_f1']:.3f}", "Quantum Accuracy", TEAL, "🎯"),
        info_card("RF F1",      f"{data['rf_f1']:.3f}", "Classical Baseline", "#636e72", "📜"),
        info_card("Q-Kernel",   f"{QSVM_QUBITS} Qubits", "RY Angle Embedding", DARK, "⧛️"),
        info_card("QAOA Ansatz", "6 Qubits", "p=1 Trotter Depth", RED, "🌀"),
    ], className="g-3 mb-4")

    # ── Tab rendering ─────────────────────────────────────────────────────
    if active_tab == "tab-qsvm":
        content = render_qsvm_tab(data, df)
    elif active_tab == "tab-qubo":
        content = render_qubo_tab(data, n_vars)
    elif active_tab == "tab-qaoa":
        content = render_qaoa_tab(data)
    elif active_tab == "tab-staff":
        content = render_staff_tab(data)
    else:
        content = render_alloc_tab(data, df)

    return summary, content


# ── QSVM tab ─────────────────────────────────────────────────────────────────

def render_qsvm_tab(data, df):
    km  = np.array(data["kernel_matrix"])
    urg = data["urgency_scores"]
    patient_ids = df["patient_id"].tolist()

    # Kernel heatmap
    fig_kernel = go.Figure(go.Heatmap(
        z=km,
        x=patient_ids, y=patient_ids,
        colorscale="YlOrRd",
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(
            tickfont=dict(color=DARK, size=10, family="Fredoka"),
            title=dict(text="OVERLAP", font=dict(color=DARK, size=10, family="Fredoka")),
        ),
    ))
    fig_kernel.update_layout(
        title=dict(text="QUANTUM KERNEL OVERLAP", font=dict(color=DARK, size=14, family="Fredoka")),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(color=DARK, gridcolor="#f1f2f6", tickangle=-45),
        yaxis=dict(color=DARK, gridcolor="#f1f2f6", autorange="reversed"),
        margin=dict(l=40, r=20, t=60, b=60),
        height=450,
    )

    # Urgency bar chart — QSVM vs RF
    rf_urg = data["rf_urgency"]
    fig_urg = go.Figure()
    fig_urg.add_trace(go.Bar(
        x=patient_ids, y=urg,
        name="Quantum (QSVM)",
        marker_color=RED,
    ))
    fig_urg.add_trace(go.Scatter(
        x=patient_ids, y=rf_urg,
        name="Classical (RF)",
        mode="markers",
        marker=dict(color=TEAL, size=12, symbol="diamond", line=dict(color=DARK, width=2)),
    ))
    fig_urg.add_hline(y=0.7, line_dash="dash", line_color=DARK, opacity=0.3,
                      annotation_text="CRITICAL", annotation_position="top left")
    
    fig_urg.update_layout(
        title=dict(text="TRIAGE: QUANTUM VS CLASSICAL", font=dict(color=DARK, size=14, family="Fredoka")),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(color=DARK, gridcolor="#f1f2f6"),
        yaxis=dict(color=DARK, gridcolor="#f1f2f6", title="URGENCY SCORE", range=[0, 1.1]),
        legend=dict(font=dict(color=DARK, size=10, family="Fredoka"), bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=40, r=20, t=60, b=60),
        height=450,
    )

    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_kernel, config={"displayModeBar": False})), className="command-card"), width=7),
            dbc.Col([
                dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_urg, config={"displayModeBar": False})), className="command-card mb-4"),
                dbc.Card(dbc.CardBody([
                    html.Div("🔬 INNER WORKINGS", style={"fontWeight": "700", "fontSize": "0.85rem", "marginBottom": "10px"}),
                    html.Small(f"We map patient vitals into a {QSVM_QUBITS}-qubit Hilbert space using Angle Embedding. The overlap (kernel) measures similarity in a way classical SVMs can't easily see.",
                               style={"color": "#636e72", "lineHeight": "1.4", "display": "block"})
                ]), className="command-card")
            ], width=5),
        ])
    ])

    # F1 comparison bar
    fig_f1 = go.Figure(go.Bar(
        x=["QSVM", "Random Forest"],
        y=[data["qsvm_f1"], data["rf_f1"]],
        marker_color=[PURPLE, GREEN],
        text=[f"{data['qsvm_f1']:.3f}", f"{data['rf_f1']:.3f}"],
        textposition="outside",
        textfont=dict(color="#e2e8f0"),
    ))
    fig_f1.update_layout(
        title=dict(text="F1 Score Comparison (macro, train set)",
                   font=dict(color="#e2e8f0", size=13)),
        paper_bgcolor="#030712", plot_bgcolor="#0a0a1a",
        xaxis=dict(color="#64748b"),
        yaxis=dict(color="#64748b", gridcolor="#1e1b4b", range=[0, 1.15]),
        margin=dict(l=40, r=20, t=50, b=40),
        height=280,
    )

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_kernel, config={"displayModeBar": False}), width=7),
            dbc.Col([
                dcc.Graph(figure=fig_f1, config={"displayModeBar": False}),
                dbc.Card(dbc.CardBody([
                    html.Div("Quantum Kernel Circuit", style={"color": "#94a3b8",
                                                               "fontSize": "0.75rem",
                                                               "textTransform": "uppercase",
                                                               "letterSpacing": "0.08em"}),
                    html.Div("4-qubit PennyLane · default.qubit simulator",
                             style={"color": PURPLE, "fontSize": "0.85rem", "marginTop": "4px"}),
                    html.Pre(
                        "Φ: Ry(πxᵢ) ⊗⁴  →  CNOT chain  →  Ry(-πxᵢ)\n"
                        "K(i,j) = |⟨0|U(xᵢ)†U(xⱼ)|0⟩|²",
                        style={"color": "#64748b", "fontSize": "0.75rem",
                               "marginTop": "6px", "marginBottom": 0},
                    ),
                ]), style={"border": f"1px solid {PURPLE}33", "background": "#0a0a1a"}),
            ], width=5),
        ], className="mb-3"),
        dcc.Graph(figure=fig_urg, config={"displayModeBar": False}),
    ])


# ── QUBO tab ──────────────────────────────────────────────────────────────────

def render_qubo_tab(data, n_vars):
    alpha = data["alpha"]

    # Reconstruct matrix from JSON-serialised dict
    raw   = data["qubo_dict"]
    M     = np.zeros((n_vars, n_vars))
    for key_str, val in raw.items():
        key_str = key_str.strip("()")
        parts   = [p.strip() for p in key_str.split(",")]
        if len(parts) == 2:
            i, j = int(parts[0]), int(parts[1])
            if 0 <= i < n_vars and 0 <= j < n_vars:
                M[i, j] += val
                if i != j:
                    M[j, i] += val

    resource_labels = ["ICU", "Vent", "Ward"]
    n_patients = n_vars // 3
    labels = [f"P{p+1:02d}-{resource_labels[r]}" for p in range(n_patients) for r in range(3)]

    fig_qubo = go.Figure(go.Heatmap(
        z=M,
        x=labels, y=labels,
        colorscale="RdBu",
        zmid=0,
        showscale=True,
        colorbar=dict(
            tickfont=dict(color=DARK, family="Fredoka"),
            title=dict(text="Q[i,j]", font=dict(color=DARK, family="Fredoka")),
        ),
    ))
    fig_qubo.update_layout(
        title=dict(text="QUBO COST LANDSCAPE (RED=PENALTY, BLUE=REWARD)",
                   font=dict(color=DARK, size=14, family="Fredoka")),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(color=DARK, gridcolor="#f1f2f6", tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(color=DARK, gridcolor="#f1f2f6", autorange="reversed", tickfont=dict(size=9)),
        margin=dict(l=80, r=20, t=60, b=100),
        height=480,
    )

    # Diagonal values bar
    diag_vals = [M[i, i] for i in range(min(n_vars, len(labels)))]
    fig_diag = go.Figure(go.Bar(
        x=labels, y=diag_vals,
        marker_color=[TEAL if v < 0 else RED for v in diag_vals],
    ))
    fig_diag.update_layout(
        title=dict(text="QUBO DIAGONAL (UTILITY vs PENALTY)",
                   font=dict(color=DARK, size=14, family="Fredoka")),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(color=DARK, tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(color=DARK, gridcolor="#f1f2f6", title="Q[i,i]"),
        margin=dict(l=60, r=20, t=50, b=80),
        height=280,
    )

    constraint_cards = dbc.Row([
        dbc.Col(info_card("α PENALTY", f"{alpha:.2f}",
                          "Enforces one-bed-per-patient", RED), width=3),
        dbc.Col(info_card("β PENALTY", "15.00",
                          "Prevents ward overcrowding", YELLOW), width=3),
        dbc.Col(info_card("HAMILTONIAN", str(n_vars),
                          f"{n_vars//3} Patients x 3 Resources"), width=3),
        dbc.Col(info_card("SOLVER", "SA (NEAL)",
                          "Simulated Annealing Samples", TEAL), width=3),
    ], className="g-3 mb-4")

    return html.Div([
        constraint_cards,
        dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_qubo, config={"displayModeBar": False})), className="command-card mb-4"),
        dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_diag, config={"displayModeBar": False})), className="command-card"),
    ])


# ── QAOA tab ─────────────────────────────────────────────────────────────────

def render_qaoa_tab(data):
    b64_img = data.get("qaoa_circuit_b64")
    info    = data.get("qaoa_info", {})
    
    if not b64_img:
        return dbc.Alert("Circuit generator idle. Run optimization first!", color="warning")

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div("QUANTUM APPROX OPTIMIZATION", className="section-header"),
                html.P([
                    "We map the hospital constraints to an Ising Hamiltonian and solve it using an alternating operator ansatz."
                ], style={"color": "#636e72", "fontSize": "0.95rem"}),
                
                dbc.Card(dbc.CardBody([
                    html.Div("CONFIGURATION", style={"fontWeight": "700", "fontSize": "0.85rem", "marginBottom": "10px"}),
                    html.Div([
                        html.Small("STEPS (P)", style={"fontWeight": "700", "color": "#b2bec3"}),
                        html.Div(f"{info.get('qaoa_p', 1)}", style={"fontSize": "1.2rem", "fontWeight": "700", "color": RED}),
                        
                        html.Small("DEMO QUBITS", style={"fontWeight": "700", "color": "#b2bec3", "marginTop": "10px", "display": "block"}),
                        html.Div(f"{info.get('demo_qubits', 6)}", style={"fontSize": "1.2rem", "fontWeight": "700", "color": DARK}),
                        
                        html.Small("TARGET", style={"fontWeight": "700", "color": "#b2bec3", "marginTop": "10px", "display": "block"}),
                        html.Div("IBM EAGLE (SIM)", style={"fontSize": "1rem", "fontWeight": "700", "color": TEAL}),
                    ])
                ]), className="command-card mb-4"),
            ], width=4),
            
            dbc.Col([
                html.Div("GENERATED ANSATZ CIRCUIT", className="section-header"),
                html.Div([
                    html.Img(src=b64_img if b64_img.startswith("data:") else f"data:image/png;base64,{b64_img}", 
                            style={"maxWidth": "100%", "height": "auto", "filter": "contrast(1.1)"})
                ], className="qaoa-circuit-box"),
                html.Small("Showing p=1 depth decomposition for a subset of the Hilbert space.",
                           style={"color": "#b2bec3", "display": "block", "marginTop": "15px", "textAlign": "center", "fontWeight": "600"})
            ], width=8),
        ])
    ])


# ── Allocation tab ────────────────────────────────────────────────────────────

def render_alloc_tab(data, df):
    q_alloc = data["q_alloc"]
    c_alloc = data["c_alloc"]
    q_util  = data["q_util"]
    c_util  = data["c_util"]

    def alloc_table(alloc, title, color):
        rows = []
        for a in alloc:
            idx  = a["patient_idx"]
            pid  = df.loc[idx, "patient_id"] if idx < len(df) else f"P{idx}"
            rows.append(html.Tr([
                html.Td(pid, style={"fontWeight": "700", "fontFamily": "JetBrains Mono"}),
                html.Td(f"{a['urgency']:.3f}", style={"color": color, "fontWeight": "700"}),
                html.Td(a["resource_name"], style={"fontWeight": "600"}),
            ]))
        return dbc.Card([
            dbc.CardHeader(html.Span(title, style={"color": "white", "fontWeight": "700"}),
                           style={"background": DARK, "padding": "12px"}),
            dbc.CardBody(dbc.Table(
                [html.Thead(html.Tr([
                    html.Th("PATIENT"), html.Th("SCORE"), html.Th("RESOURCE"),
                ])),
                 html.Tbody(rows)],
                bordered=False, hover=True, responsive=True, className="table",
            )),
        ], className="command-card")

    def pie_chart(vals, title, colors):
        fig = go.Figure(go.Pie(
            labels=list(vals.keys()), values=list(vals.values()),
            hole=0.6,
            marker=dict(colors=colors, line=dict(color=DARK, width=3)),
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(color=DARK, size=13, family="Fredoka")),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(font=dict(color=DARK, family="Fredoka"), bgcolor="rgba(255,255,255,0.5)"),
            margin=dict(l=20, r=20, t=50, b=20),
            height=280,
        )
        return fig

    q_colors = [RED, YELLOW, TEAL]
    fig_q_pie = pie_chart(q_util, "QUANTUM UTILIZATION", q_colors)
    fig_c_pie = pie_chart(c_util, "CLASSICAL UTILIZATION", [c + "88" for c in q_colors])

    q_map = {a["patient_idx"]: a["resource_name"] for a in q_alloc}
    c_map = {a["patient_idx"]: a["resource_name"] for a in c_alloc}
    diff_rows = []
    for idx, row in df.iterrows():
        q_r = q_map.get(idx, "—")
        c_r = c_map.get(idx, "—")
        changed = q_r != c_r
        diff_rows.append(html.Tr([
            html.Td(row["patient_id"], style={"fontWeight": "700"}),
            html.Td(q_r, style={"color": RED if not changed else TEAL, "fontWeight": "700"}),
            html.Td(c_r, style={"color": "#636e72"}),
            html.Td(dbc.Badge("ROTATED" if changed else "MATCHED", 
                              color="warning" if changed else "success",
                              style={"borderRadius": "8px"})),
        ]))

    return html.Div([
        dbc.Row([
            dbc.Col(alloc_table(q_alloc, "⚛  QUANTUM ALLOCATION", RED), width=6),
            dbc.Col(alloc_table(c_alloc, "🔢  CLASSICAL GREEDY", DARK),  width=6),
        ], className="g-4 mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_q_pie, config={"displayModeBar": False})), className="command-card"), width=6),
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_c_pie, config={"displayModeBar": False})), className="command-card"), width=6),
        ], className="g-4 mb-4"),
        dbc.Card([
            dbc.CardHeader("ALLOCATION SHIFTS (Q vs C)", style={"background": DARK, "color": "white", "fontWeight": "700"}),
            dbc.CardBody(dbc.Table(
                [html.Thead(html.Tr([
                    html.Th("PATIENT"), html.Th("QUANTUM"), html.Th("CLASSICAL"), html.Th("STATUS"),
                ])),
                 html.Tbody(diff_rows)],
                bordered=False, hover=True, responsive=True, className="table",
            )),
        ], className="command-card"),
    ])



# ── Staff QUBO tab ────────────────────────────────────────────────────────────

def render_staff_tab(data):
    import io as _io
    sm        = data.get("staff_metrics", {})
    s_alloc   = data.get("staff_allocation", [])
    alpha_s   = data.get("alpha_s", 0)
    s1_ms     = data.get("stage1_solve_ms", "?")
    s2_ms     = data.get("stage2_solve_ms", "?")
    staff_df  = pd.read_json(_io.StringIO(data["staff_df"]), orient="split") if "staff_df" in data else None
    n_staff   = len(staff_df) if staff_df is not None else 0
    n_patients = data.get("n_patients", 8)
    n_vars_s   = n_staff * n_patients

    if staff_df is None or not s_alloc:
        return dbc.Alert("Run the pipeline to generate Stage 2 staff data.", color="warning")

    # ── Staff QUBO heatmap ────────────────────────────────────────────────
    raw_s      = data.get("staff_qubo_dict", {})
    is_greedy  = bool(raw_s.get("__greedy__", False))
    QUBO_LIMIT = 400   # mirrors STAFF_QUBO_VAR_LIMIT in staff_optimizer.py
    max_pts_for_qubo = QUBO_LIMIT // max(n_staff, 1)

    if is_greedy:
        fig_sq = go.Figure()
        fig_sq.add_annotation(
            text=(
                f"<b>Greedy fallback — QUBO matrix unavailable</b><br>"
                f"{n_vars_s:,} decision variables ({n_staff} staff × {n_patients} patients)<br>"
                f"exceeds the {QUBO_LIMIT}-variable QUBO limit.<br><br>"
                f"<span style='color:#636e72'>Use ≤ {max_pts_for_qubo} patients to see the full matrix.</span>"
            ),
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, align="center",
            font=dict(size=14, color=DARK, family="Fredoka"),
        )
        fig_sq.update_layout(
            paper_bgcolor="white", plot_bgcolor="#f8f9fa",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            title=dict(
                text=f"STAGE 2 STAFF QUBO  ({n_vars_s} VARS > {QUBO_LIMIT} LIMIT — GREEDY USED)",
                font=dict(color=RED, size=13, family="Fredoka"),
            ),
            height=300,
        )
    else:
        M_s = np.zeros((n_vars_s, n_vars_s))
        for key_str, val in raw_s.items():
            key_str = key_str.strip("()")
            parts   = [p.strip().strip("'\"") for p in key_str.split(",")]
            if len(parts) == 2:
                def _vidx(v):
                    parts_v = v.split("_p")
                    return int(parts_v[0][1:]) * n_patients + int(parts_v[1])
                try:
                    i, j = _vidx(parts[0]), _vidx(parts[1])
                    if 0 <= i < n_vars_s and 0 <= j < n_vars_s:
                        M_s[i, j] += val
                        if i != j:
                            M_s[j, i] += val
                except (IndexError, ValueError):
                    pass

        DISPLAY  = min(n_vars_s, 48)
        M_disp   = M_s[:DISPLAY, :DISPLAY]
        role_labels = staff_df["role_name"].tolist()
        xlabels = [
            f"{role_labels[n_idx][:3]}-P{p_idx+1}"
            for n_idx in range(min(n_staff, DISPLAY // n_patients + 1))
            for p_idx in range(n_patients)
        ][:DISPLAY]

        fig_sq = go.Figure(go.Heatmap(
            z=M_disp, x=xlabels, y=xlabels,
            colorscale="RdBu", zmid=0,
            showscale=True,
            colorbar=dict(
                tickfont=dict(color=DARK, family="Fredoka"),
                title=dict(text="Q_s[i,j]", font=dict(color=DARK, family="Fredoka")),
            ),
        ))
        fig_sq.update_layout(
            title=dict(text=f"STAGE 2 STAFF QUBO  ({n_vars_s} VARS \u2014 SHOWING FIRST {DISPLAY})",
                       font=dict(color=DARK, size=13, family="Fredoka")),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(color=DARK, tickangle=45, tickfont=dict(size=7)),
            yaxis=dict(color=DARK, autorange="reversed", tickfont=dict(size=7)),
            margin=dict(l=80, r=20, t=60, b=100),
            height=480,
        )

    # ── Staff utilisation bar chart ───────────────────────────────────────
    util_pct = sm.get("utilization_pct", {})
    fig_util = go.Figure(go.Bar(
        x=list(util_pct.keys()),
        y=list(util_pct.values()),
        marker_color=[TEAL if v <= 80 else RED for v in util_pct.values()],
        text=[f"{v}%" for v in util_pct.values()],
        textposition="outside",
        textfont=dict(family="Fredoka", color=DARK),
    ))
    fig_util.add_hline(y=80, line_dash="dash", line_color=DARK, opacity=0.3,
                       annotation_text="80% FULL", annotation_position="top right")
    fig_util.update_layout(
        title=dict(text="STAFF UTILISATION % PER ROLE", font=dict(color=DARK, size=13, family="Fredoka")),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(color=DARK),
        yaxis=dict(color=DARK, gridcolor="#f1f2f6", range=[0, 115], title="Utilisation (%)"),
        margin=dict(l=50, r=20, t=50, b=60),
        height=300,
    )

    # ── Skill-acuity scatter ──────────────────────────────────────────────
    fig_sam = go.Figure()
    if s_alloc:
        fig_sam.add_trace(go.Scatter(
            x=[a["urgency"] for a in s_alloc],
            y=[a["skill_level"] for a in s_alloc],
            mode="markers+text",
            text=[a["staff_id"] for a in s_alloc],
            textposition="top center",
            textfont=dict(size=9, family="Fredoka"),
            marker=dict(
                size=14,
                color=[a["fatigue_score"] for a in s_alloc],
                colorscale="RdYlGn_r",
                showscale=True,
                colorbar=dict(title="Fatigue", thickness=10),
                line=dict(color=DARK, width=2),
            ),
        ))
    fig_sam.update_layout(
        title=dict(text="SKILL-ACUITY MATCH  (colour = fatigue)", font=dict(color=DARK, size=13, family="Fredoka")),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(color=DARK, gridcolor="#f1f2f6", title="Patient Urgency"),
        yaxis=dict(color=DARK, gridcolor="#f1f2f6", title="Staff Skill Level"),
        margin=dict(l=60, r=20, t=60, b=60),
        height=320,
    )

    # ── Metric strip ─────────────────────────────────────────────────────
    metric_strip = dbc.Row([
        info_card("S2 QUBITS",     str(n_vars_s),       f"{n_staff} staff \u00d7 {n_patients} pts", DARK, "\ud83e\udde0"),
        info_card("\u03b1_s PENALTY",    f"{alpha_s:.1f}",    "Uniqueness dominance", RED, "\u2696\ufe0f"),
        info_card("SKILL MATCH",  f"{sm.get('skill_acuity_match', 0):.3f}", "Skill \u00d7 Acuity avg", TEAL, "\ud83c\udfaf"),
        info_card("UNASSIGNED",   str(sm.get("unassigned_count", "?")), "Patients w/o staff", RED if sm.get("unassigned_count", 1) > 0 else TEAL, "\u26a0\ufe0f"),
        info_card("CROSS-QUAL",   f"{sm.get('cross_qual_rate', 0)}%", "Floated staff rate", YELLOW, "\ud83d\udd04"),
        info_card("SOLVE TIMES",  f"{s1_ms}/{s2_ms}ms", "Stage1 \u2192 Stage2", DARK, "\u23f1\ufe0f"),
    ], className="g-3 mb-4")

    return html.Div([
        metric_strip,
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_sq, config={"displayModeBar": False})), className="command-card"), width=7),
            dbc.Col([
                dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_util, config={"displayModeBar": False})), className="command-card mb-4"),
                dbc.Card(dbc.CardBody(dcc.Graph(figure=fig_sam, config={"displayModeBar": False})), className="command-card"),
            ], width=5),
        ], className="g-4"),
    ])


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, port=8051, host="0.0.0.0")
