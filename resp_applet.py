import numpy as np
import plotly.graph_objects as gg
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="Medizintechnik: Beatmungsmodi", layout="wide"
)

st.title("Volumengesteuerte (VCV) vs. Druckgesteuerte (PCV) Beatmung")

# --- SIDEBAR: Parameter ---
st.sidebar.header("Lungenmechanik")
compliance = st.sidebar.slider(
    "Compliance C (mL/mbar)",
    min_value=10,
    max_value=100,
    value=50,
    step=5,
    help="Niedrig = Steife Lunge (z.B. ARDS), Hoch = Dehnbare Lunge",
)

# Resistance ist fest auf einen realistischen Grundwert gesetzt
resistance = 5.0  # mbar / (L/s)

st.sidebar.header("Beatmungseinstellungen")
peep = st.sidebar.number_input("PEEP (mbar)", min_value=0, max_value=20, value=5)
freq = st.sidebar.number_input(
    "Atemfrequenz (1/min)", min_value=5, max_value=40, value=15
)
mode = st.sidebar.radio(
    "Beatmungsmodus", ["Volumengesteuert (VCV)", "Druckgesteuert (PCV)"]
)

if mode == "Volumengesteuert (VCV)":
    v_t_target = st.sidebar.slider(
        "Tidalvolumen V_T (mL)", min_value=200, max_value=800, value=500, step=50
    )
    p_insp_target = None
else:
    p_insp_target = st.sidebar.slider(
        "Inspirationsdruck p_insp (mbar)",
        min_value=10,
        max_value=40,
        value=20,
        step=1,
    )
    v_t_target = None

# --- BERECHNUNG DER KURVEN ---
# Zeitachse für einen Atemzug
t_cycle = 60.0 / freq
i_e_ratio = 1 / 2  # 1:2 Verhaltnis
t_insp = t_cycle * (i_e_ratio / (1 + i_e_ratio))
t_exp = t_cycle - t_insp

t = np.linspace(0, t_cycle, 500)
p_t = np.zeros_like(t)
flow_t = np.zeros_like(t)
v_t = np.zeros_like(t)

c_l = compliance / 1000.0  # Umrechnung in L/mbar
tau = resistance * c_l  # Zeitkonstante

for i, ti in enumerate(t):
    if ti <= t_insp:
        # Inspirationsphase
        if mode == "Volumengesteuert (VCV)":
            # Konstantflow
            flow_l_s = (v_t_target / 1000.0) / t_insp
            flow_t[i] = flow_l_s * 60.0  # in L/min
            v_t[i] = flow_l_s * ti * 1000.0  # in mL
            p_t[i] = peep + (v_t[i] / compliance) + (resistance * flow_l_s)
        else:
            # Druckgesteuert (PCV)
            delta_p = p_insp_target - peep
            p_t[i] = p_insp_target
            flow_l_s = (delta_p / resistance) * np.exp(-ti / tau)
            flow_t[i] = flow_l_s * 60.0  # in L/min
            v_t[i] = c_l * delta_p * (1 - np.exp(-ti / tau)) * 1000.0  # in mL
    else:
        # Exspirationsphase (passiver Abfall)
        te = ti - t_insp
        if i > 0:
            v_end_insp = v_t[np.where(t <= t_insp)[0][-1]] / 1000.0
        else:
            v_end_insp = 0

        flow_l_s = -(v_end_insp / tau) * np.exp(-te / tau)
        flow_t[i] = flow_l_s * 60.0
        v_t[i] = v_end_insp * np.exp(-te / tau) * 1000.0
        p_t[i] = peep + (v_t[i] / compliance) + (resistance * flow_l_s)

# --- GRAPHEN ERSTELLEN ---
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        "Druck p(t) [mbar]",
        "Flow V'(t) [L/min]",
        "Volumen V(t) [mL]",
    ),
)

fig.add_trace(
    gg.Scatter(x=t, y=p_t, name="Druck", line=dict(color="red", width=2)),
    row=1,
    col=1,
)
fig.add_trace(
    gg.Scatter(x=t, y=flow_t, name="Flow", line=dict(color="blue", width=2)),
    row=2,
    col=1,
)
fig.add_trace(
    gg.Scatter(x=t, y=v_t, name="Volumen", line=dict(color="green", width=2)),
    row=3,
    col=1,
)

fig.update_layout(height=600, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
fig.update_xaxes(title_text="Zeit [s]", row=3, col=1)

# --- DASHBOARD LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Messwerte & Risiken")

    max_p = np.max(p_t)
    max_v = np.max(v_t)

    st.metric("Spitzendruck (p_peak)", f"{max_p:.1f} mbar")
    st.metric("Erzieltes Tidalvolumen (V_T)", f"{max_v:.0f} mL")

    st.markdown("---")

    # Warnanzeigen
    if max_p > 30:
        st.error("⚠️ **BAROTRAUMA-RISIKO!** Der Druck überschreitet 30 mbar.")
    elif max_p <= 30:
        st.success("✅ Druck im sicheren Bereich (≤ 30 mbar).")

    if max_v < 300:
        st.warning(
            "⚠️ **HYPOVENTILATION!** Das Atemzugvolumen ist kritisch niedrig."
        )
    elif max_v > 700:
        st.warning("⚠️ **VOLUTRAUMA-RISIKO!** Das Atemzugvolumen ist sehr hoch.")
