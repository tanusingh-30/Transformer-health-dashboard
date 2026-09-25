"""
Transformer Health Monitor
Data -> Pandas -> Calculations -> Plotly -> Streamlit

Run:
    pip install streamlit pandas numpy plotly
    streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Transformer Health Monitor", layout="wide", page_icon="⚡")

REQUIRED_COLS = [
    "timestamp", "transformer_id", "temperature", "ambient_temperature",
    "current", "voltage", "humidity", "load",
]

# ----------------------------------------------------------------------------
# DATA LAYER
# ----------------------------------------------------------------------------
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=["timestamp"])
    return df


def validate(df: pd.DataFrame) -> list[str]:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return missing


# ----------------------------------------------------------------------------
# SIDEBAR — data source + editable thresholds (this is the "editable" part)
# ----------------------------------------------------------------------------
st.sidebar.title("⚡ Controls")

st.sidebar.subheader("1. Data source")
uploaded = st.sidebar.file_uploader("Upload sensor CSV", type=["csv"])

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None

if uploaded is not None:
    df_loaded = load_csv(uploaded)
    missing = validate(df_loaded)
    if missing:
        st.sidebar.error(f"CSV is missing columns: {missing}")
        st.stop()
    st.session_state.raw_df = df_loaded
elif st.session_state.raw_df is None:
    st.sidebar.info("No file uploaded — using bundled sample_transformer_data.csv if present, else demo data.")
    try:
        st.session_state.raw_df = pd.read_csv("sample_transformer_data.csv", parse_dates=["timestamp"])
    except FileNotFoundError:
        # tiny inline fallback so the app never crashes with zero data
        rng = np.random.default_rng(0)
        n = 200
        ts = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="5min")
        st.session_state.raw_df = pd.DataFrame({
            "timestamp": ts,
            "transformer_id": "TXF-001",
            "temperature": 60 + rng.normal(0, 3, n).cumsum() * 0.05 + 20,
            "ambient_temperature": 30 + rng.normal(0, 1, n),
            "current": 70 + rng.normal(0, 5, n),
            "voltage": 415 + rng.normal(0, 2, n),
            "humidity": 55 + rng.normal(0, 5, n),
            "load": 65 + rng.normal(0, 8, n),
        })

# Downloadable CSV template for connecting real sensor data later
template_df = pd.DataFrame({
    "timestamp": pd.date_range("2026-09-25 10:00", periods=5, freq="5min"),
    "transformer_id": ["TXF-001"] * 5,
    "temperature": [61.0, 63.5, 66.0, 69.0, 72.0],
    "ambient_temperature": [30.0, 30.5, 31.0, 31.0, 31.5],
    "current": [52.0, 56.0, 60.0, 64.0, 68.0],
    "voltage": [415.0, 414.5, 415.2, 413.8, 414.7],
    "humidity": [55.0, 56.0, 57.0, 58.0, 58.0],
    "load": [52.0, 56.0, 60.0, 64.0, 68.0],
})
st.sidebar.download_button(
    "⬇️ Download CSV template",
    template_df.to_csv(index=False),
    file_name="transformer_sensor_template.csv",
    mime="text/csv",
)

st.sidebar.subheader("2. Transformer")
tid_list = sorted(st.session_state.raw_df["transformer_id"].unique())
selected_tid = st.sidebar.selectbox("Select transformer", tid_list)

st.sidebar.subheader("3. Rated limits (editable)")
max_temp = st.sidebar.slider("Max safe temperature (°C)", 40, 150, 90)
max_current = st.sidebar.slider("Rated current (A)", 10, 300, 100)
max_load_pct = st.sidebar.slider("Rated load (%)", 50, 150, 100)
rapid_rise_threshold = st.sidebar.slider("Rapid rise alert (°C per reading, last 5 avg)", 0.1, 5.0, 1.0, 0.1)

st.sidebar.subheader("4. Prediction")
horizon = st.sidebar.slider("Forecast horizon (readings ahead)", 1, 50, 12)

# ----------------------------------------------------------------------------
# EDITABLE DATA TABLE
# ----------------------------------------------------------------------------
with st.expander("📝 Edit raw sensor data (add rows, fix values, then Apply)"):
    tf_mask = st.session_state.raw_df["transformer_id"] == selected_tid
    editable_slice = st.session_state.raw_df[tf_mask].reset_index(drop=True)
    edited = st.data_editor(
        editable_slice,
        num_rows="dynamic",
        use_container_width=True,
        key=f"editor_{selected_tid}",
    )
    if st.button("Apply edits"):
        other = st.session_state.raw_df[~tf_mask]
        st.session_state.raw_df = pd.concat([other, edited], ignore_index=True)
        st.success("Data updated.")
        st.rerun()

df = st.session_state.raw_df[st.session_state.raw_df["transformer_id"] == selected_tid].copy()
df = df.sort_values("timestamp").reset_index(drop=True)

if df.empty or len(df) < 2:
    st.warning("Not enough rows for this transformer to compute anything meaningful.")
    st.stop()

# ----------------------------------------------------------------------------
# CALCULATIONS (Pandas)
# ----------------------------------------------------------------------------
df["temperature_rise"] = df["temperature"] - df["ambient_temperature"]
df["load_pct"] = df["load"]
df["temp_roll5"] = df["temperature"].rolling(5, min_periods=1).mean()
df["temp_rate_of_change"] = df["temperature"].diff().rolling(5, min_periods=1).mean()

latest = df.iloc[-1]
prev = df.iloc[-2]


def compute_health_score(row) -> float:
    score = 100.0
    if row["temperature"] > max_temp:
        score -= min(40.0, (row["temperature"] - max_temp) * 2.0)
    if row["current"] > max_current:
        score -= min(30.0, (row["current"] - max_current) * 1.5)
    if row["load"] > max_load_pct:
        score -= min(30.0, (row["load"] - max_load_pct) * 1.0)
    if row["humidity"] > 85:
        score -= min(10.0, (row["humidity"] - 85) * 0.5)
    return round(max(0.0, min(100.0, score)), 1)


health_score = compute_health_score(latest)
prev_health_score = compute_health_score(prev)

# Linear-trend prediction: fit temperature vs index over last N points, extrapolate `horizon` steps ahead
window = df.tail(30)
if len(window) >= 3:
    x = np.arange(len(window))
    coeffs = np.polyfit(x, window["temperature"].values, 1)
    slope, intercept = coeffs[0], coeffs[1]
    predicted_temp = float(np.polyval(coeffs, len(window) - 1 + horizon))
else:
    slope = 0.0
    predicted_temp = float(latest["temperature"])

if predicted_temp >= max_temp:
    risk = "High"
    risk_color = "#e74c3c"
elif predicted_temp >= max_temp * 0.9:
    risk = "Moderate"
    risk_color = "#f39c12"
else:
    risk = "Low"
    risk_color = "#2ecc71"

# ----------------------------------------------------------------------------
# ALERTS (rule-based)
# ----------------------------------------------------------------------------
alerts = []
if latest["temp_rate_of_change"] > rapid_rise_threshold:
    alerts.append(("⚠️", f"Temperature rising rapidly (+{latest['temp_rate_of_change']:.2f}°C/reading avg)"))
if latest["temperature"] > max_temp:
    alerts.append(("🔥", f"Temperature {latest['temperature']:.1f}°C exceeds rated limit {max_temp}°C"))
if latest["current"] > max_current:
    alerts.append(("🔌", f"Overcurrent: {latest['current']:.1f}A exceeds rated {max_current}A"))
if latest["load"] > max_load_pct:
    alerts.append(("📈", f"Overload: {latest['load']:.1f}% exceeds rated {max_load_pct}%"))
if latest["humidity"] > 85:
    alerts.append(("💧", f"High humidity ({latest['humidity']:.1f}%) — insulation degradation risk"))
if health_score < 50:
    alerts.append(("🚨", f"Critical health score ({health_score}/100) — inspect immediately"))
if not alerts:
    alerts.append(("✅", "All parameters within normal range"))

# ----------------------------------------------------------------------------
# LAYOUT (Streamlit + Plotly)
# ----------------------------------------------------------------------------
st.title("⚡ TRANSFORMER HEALTH MONITOR")
st.subheader(f"Transformer {selected_tid}")
st.caption(f"Last reading: {latest['timestamp']}")

st.divider()

# --- top metrics row ---
c1, c2, c3 = st.columns(3)
c1.metric("HEALTH", f"{health_score}/100", delta=round(health_score - prev_health_score, 1))
c2.metric("TEMPERATURE", f"{latest['temperature']:.1f}°C", delta=round(latest['temperature'] - prev['temperature'], 1))
c3.metric("CURRENT", f"{latest['current']:.1f}A", delta=round(latest['current'] - prev['current'], 1))

st.divider()

# --- temperature vs time chart ---
st.markdown("#### TEMPERATURE vs TIME")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["temperature"], mode="lines", name="Temperature",
                          line=dict(color="#e74c3c", width=2)))
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["temp_roll5"], mode="lines", name="5-pt rolling avg",
                          line=dict(color="#3498db", width=1, dash="dot")))
fig.add_hline(y=max_temp, line_dash="dash", line_color="orange",
              annotation_text=f"Rated limit {max_temp}°C", annotation_position="top left")
fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                   xaxis_title=None, yaxis_title="°C", legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- electrical / environment two-column section ---
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### ELECTRICAL")
    st.metric("Voltage", f"{latest['voltage']:.1f} V", delta=round(latest['voltage'] - prev['voltage'], 1))
    st.metric("Current", f"{latest['current']:.1f} A", delta=round(latest['current'] - prev['current'], 1))
    st.metric("Load", f"{latest['load']:.1f} %", delta=round(latest['load'] - prev['load'], 1))
    fig_elec = go.Figure()
    fig_elec.add_trace(go.Scatter(x=df["timestamp"], y=df["current"], name="Current (A)", line=dict(color="#9b59b6")))
    fig_elec.add_trace(go.Scatter(x=df["timestamp"], y=df["load"], name="Load (%)", line=dict(color="#16a085")))
    fig_elec.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig_elec, use_container_width=True)

with col_right:
    st.markdown("#### ENVIRONMENT")
    st.metric("Ambient Temp", f"{latest['ambient_temperature']:.1f}°C")
    st.metric("Humidity", f"{latest['humidity']:.1f} %")
    st.metric("Temperature Rise", f"{latest['temperature_rise']:.1f}°C")
    fig_env = go.Figure()
    fig_env.add_trace(go.Scatter(x=df["timestamp"], y=df["ambient_temperature"], name="Ambient (°C)", line=dict(color="#f1c40f")))
    fig_env.add_trace(go.Scatter(x=df["timestamp"], y=df["humidity"], name="Humidity (%)", line=dict(color="#2980b9")))
    fig_env.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig_env, use_container_width=True)

st.divider()

# --- prediction ---
st.markdown("#### PREDICTION")
st.caption("Prototype trend estimate — replace with the trained ML model when real labelled data is available.")
p1, p2, p3 = st.columns(3)
p1.metric(f"Expected Temp (+{horizon} readings)", f"{predicted_temp:.1f}°C")
p2.markdown(
    f"<div style='padding:10px;border-radius:8px;background:{risk_color}22;border:1px solid {risk_color};'>"
    f"<b>Risk: <span style='color:{risk_color}'>{risk}</span></b></div>",
    unsafe_allow_html=True,
)
p3.metric("Trend slope", f"{slope:+.3f}°C/reading")

st.divider()

# --- alerts ---
st.markdown("#### ALERTS")
for icon, msg in alerts:
    st.write(f"{icon} {msg}")
