"""
ui/dashboard.py
Streamlit demo dashboard for the hackathon.

Run with:
    streamlit run ui/dashboard.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pipeline import run_pipeline
from data.generator import RESOURCE_NAMES

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QX Hospital Optimizer",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Quantum-Enhanced Emergency Hospital Resource Optimizer")
st.caption("Quantum Vibes Hackathon 2026 · Problem Statements #18 + #1 + #16")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Simulation Controls")
    aqi = st.slider(
        "🌫️ Environmental AQI (PM2.5)",
        min_value=0, max_value=500, value=50, step=10,
        help="Higher AQI → more respiratory emergencies"
    )
    n_patients = st.slider("👤 Number of Patients", 4, 12, 8, step=2)
    run_btn = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    **Stack**  
    PennyLane · scikit-learn · D-Wave neal · Streamlit  
    
    **Citations**  
    Havlíček et al., *Nature* 2019  
    Farhi et al., arXiv:1411.4028  
    Glover et al., arXiv:1811.11538  
    Preskill, *Quantum* 2018  
    """)

# ── Main area ─────────────────────────────────────────────────────────────────
if not run_btn:
    st.info("👈 Set parameters and click **Run Pipeline** to start.")
    st.stop()

with st.spinner("Computing quantum kernel matrix and optimizing allocation..."):
    results = run_pipeline(aqi_level=float(aqi), n_patients=n_patients, verbose=False)

df             = results["df"]
q_alloc        = results["quantum_allocation"]
c_alloc        = results["classical_allocation"]
urgency_scores = results["urgency_scores"]

# ── Row 1: Key metrics ────────────────────────────────────────────────────────
st.subheader("📊 Pipeline Results")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Patients", n_patients)
col2.metric("AQI Level", f"{aqi} µg/m³")
col3.metric("Unallocated (Quantum)", results["unallocated_quantum"], delta_color="inverse")
col4.metric("Unallocated (Classical)", results["unallocated_classical"], delta_color="inverse")

# ── Row 2: Patient urgency scores ─────────────────────────────────────────────
st.subheader("🔬 Quantum Urgency Scores (QSVM Output)")

urgency_df = df[["patient_id", "bp_deviation", "spo2_deficit", "aqi_pm25", "urgency_score"]].copy()
urgency_df.columns = ["Patient", "BP Deviation", "SpO₂ Deficit", "AQI Factor", "Urgency Score"]

fig_urgency = px.bar(
    urgency_df, x="Patient", y="Urgency Score",
    color="Urgency Score", color_continuous_scale="RdYlGn_r",
    range_y=[0, 1], title="Patient Risk Urgency (Quantum SVM)"
)
st.plotly_chart(fig_urgency, use_container_width=True)

with st.expander("📋 Full Patient Feature Table"):
    st.dataframe(urgency_df, use_container_width=True)

# ── Row 3: Side-by-side allocation comparison ─────────────────────────────────
st.subheader("⚖️ Allocation Comparison: Classical Greedy vs Quantum Optimized")
left, right = st.columns(2)

def alloc_to_df(alloc, df_ref):
    rows = []
    for a in alloc:
        pid = df_ref.loc[a["patient_idx"], "patient_id"] if a["patient_idx"] < len(df_ref) else f"P{a['patient_idx']}"
        rows.append({"Patient": pid, "Urgency": a["urgency"], "Assigned Resource": a["resource_name"]})
    return pd.DataFrame(rows)

with left:
    st.markdown("**🔴 Classical Greedy Baseline**")
    c_df = alloc_to_df(c_alloc, df)
    st.dataframe(c_df, use_container_width=True, hide_index=True)

with right:
    st.markdown("**🟢 Quantum Optimized Allocation**")
    q_df = alloc_to_df(q_alloc, df)
    st.dataframe(q_df, use_container_width=True, hide_index=True)

# ── Row 4: Utilization breakdown ──────────────────────────────────────────────
st.subheader("🏨 Resource Utilization")
util_col1, util_col2 = st.columns(2)

def util_pie(util_dict, title):
    fig = go.Figure(go.Pie(
        labels=list(util_dict.keys()),
        values=list(util_dict.values()),
        hole=0.4
    ))
    fig.update_layout(title_text=title, showlegend=True)
    return fig

with util_col1:
    st.plotly_chart(util_pie(results["classical_utilization"], "Classical"), use_container_width=True)
with util_col2:
    st.plotly_chart(util_pie(results["quantum_utilization"], "Quantum"), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ No quantum speedup claimed. "
    "This demo targets approximation quality and QPU portability (Ising-native). "
    "NISQ honest: Preskill, Quantum 2, 79 (2018)."
)
