import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go

from utils import (precompute_uplift_stats, compute_profit_curves,)

# ----------------------------
# Configuración básica
# ----------------------------
st.set_page_config(
    page_title="Uplift Marketing Optimizer",
    layout="wide"
)

st.title("Uplift Marketing – Profit Optimization Demo")
st.markdown(
"""
This demo shows how uplift modeling can be used to **optimize targeting decisions**
by balancing treatment cost and conversion value.
"""
)

# ----------------------------
# Cargar datos (una sola vez)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR, "..", "data", "processed",
    "criteo-uplift-processed_sample10_sorted.parquet"
)

@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)


df = load_data()


# ----------------------------
# Precomputación pesada (UNA VEZ)
# ----------------------------
PERCENTILES = np.linspace(0.01, 1.0, 100)

@st.cache_data
def precompute(df):
    return precompute_uplift_stats(df)


uplift_stats = precompute(df)

# ----------------------------
# Inputs de negocio
# ----------------------------
st.sidebar.header("Business Parameters")

cpm = st.sidebar.number_input(
    "CPM (€)",
    min_value=0.0,
    value=3.0,
    step=0.5
)

impressions_per_user = st.sidebar.number_input(
    "Impressions per user",
    min_value=1,
    value=10,
    step=1
)

conversion_value = st.sidebar.number_input(
    "Revenue per conversion (€)",
    min_value=0.0,
    value=40.0,
    step=1.0
)

population_size = st.sidebar.number_input(
    "Population size",
    min_value=100_000,
    value=1_000_000,
    step=50_000,
    format="%d"
)

# ----------------------------
# Cálculo rápido (en cada interacción)
# ----------------------------

results = compute_profit_curves(
    uplift_stats=uplift_stats,
    population_size=population_size,
    cpm=cpm,
    impressions_per_user=impressions_per_user,
    conversion_value=conversion_value
)


st.info(
    f"Baseline (no treatment): "
    f"{population_size * uplift_stats['uplift_global'] * conversion_value:,.0f} € "
    f"(shown as y = 0 in the chart)"
)


# ----------------------------
# Gráfica (Plotly)
# ----------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=results["percentiles"] * 100,
    y=results["profit_uplift"],
    mode="lines",
    name="Uplift-based targeting",
    line=dict(width=3)
))

fig.add_trace(go.Scatter(
    x=results["percentiles"] * 100,
    y=results["profit_random"],
    mode="lines",
    name="Random targeting",
    line=dict(width=3, dash="dash")
))

fig.add_trace(go.Scatter(
    x=[results["best_percentile"] * 100] * 2,
    y=[
        0,
        max(results["profit_uplift"].max(), results["profit_random"].max())
    ],
    mode="lines",
    name="Optimal percentile",
    line=dict(color="black", dash="dot")
))

fig.update_layout(
    title="Profit vs Targeted Population",
    xaxis_title="Proportion of population targeted (%)",
    yaxis_title="Expected profit (€)",
    template="plotly_white",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Resultados clave
# ----------------------------
st.markdown("### Optimal Strategy")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Optimal percentile",
    f"{results['best_percentile']*100:.1f}%"
)

col2.metric(
    "Profit (uplift)",
    f"{results['best_profit']:,.0f} €"
)

col3.metric(
    "Profit (random)",
    f"{results['random_profit_at_best']:,.0f} €"
)
