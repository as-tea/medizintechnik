import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="Medizintechnik: Beatmungsmodi", layout="wide"
)

st.title("Volumengesteuerte (VCV) vs. Druckgesteuerte (PCV) Beatmung")

# --- 5. ARBEITSAUFTRÄGE (Aufklappbare Box) ---
with st.expander("📋 **Arbeitsaufträge für Studierende (Hier klicken zum Aufklappen)**"):
    st.markdown("""
    **1. Das ARDS-Experiment (Veränderung der Compliance):**
    * Wähle zuerst den **VCV-Modus** ($V_T = 500\\text{ mL}$). Senke die Compliance schrittweise von $60\\text{ mL/mbar}$ auf $20\\text{ mL/mbar}$. Beobachte, was mit dem Spitzendruck ($p_{\\text{peak}}$) passiert.
    * Wechsle nun in den **PCV-Modus** ($p_{\\text{insp}} = 20\\text{ mbar}$). Senke die Compliance erneut auf $20\\text{ mL/mbar}$. Was passiert hier mit dem erzielten Atemzugvolumen ($V_T$)?

    **2. Das Barotrauma-Risiko (VCV):**
    * Finde im VCV-Modus bei einer sehr steifen Lunge ($C = 20\\text{ mL/mbar}$) ein Tidalvolumen, bei dem die Sicherheitsgrenze von $30\\text{ mbar}$ gerade noch **nicht** überschritten wird.

    **3. Zielvolumen sichern (PCV):**
    * Versuche im PCV-Modus bei $C = 30\\text{ mL/mbar}$ ein Tidalvolumen von $500\\text{ mL}$ zu erreichen. Wie hoch musst du den Inspirationsdruck einstellen? Überschreitest du dabei die Druckgrenze?
    """)

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

resistance = 5.0  # Festwert in mbar / (L/s)

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
t_cycle = 60.0 / freq
i_e_ratio = 1 / 2  # 1:2 Verhältnis
t_insp = t_cycle * (i_e_ratio / (1 + i_e_ratio))

t = np.linspace(0, t_cycle, 500)
p_t = np.zeros_like(t)
flow_t = np.zeros_like(t)
v_t = np.zeros_like(t)

c_l = compliance / 1000.0
tau = resistance * c_l

for i, ti in enumerate(t):
    if ti <= t_insp:
        if mode == "Volumengesteuert (VCV)":
            flow_l_s = (v_t_target / 1000.0) / t_insp
            flow_t[i] = flow_l_s * 60.0
            v_t[i] = flow_l_s * ti * 1000.0
            p_t[i] = peep + (v_t[i] / compliance) + (resistance * flow_l_s)
        else:
            delta_p = p_insp_target - peep
            p_t[i] = p_insp_target
            flow_l_s = (delta_p / resistance) * np.exp(-ti / tau)
            flow_t[i] = flow_l_s * 60.0
            v_t[i] = c_l * delta_p * (1 - np.exp(-ti / tau)) * 1000.0
    else:
        te = ti - t_insp
        v_end_insp = v_t[np.where(t <= t_insp)[0][-1]] / 1000.0

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
    go.Scatter(x=t, y=p_t, name="Druck", line=dict(color="blue", width=2)),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=t, y=flow_t, name="Flow", line=dict(color="teal", width=2)),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(x=t, y=v_t, name="Volumen", line=dict(color="green", width=2)),
    row=3,
    col=1,
)

# 2) Rote Linie bei 30 mbar (Sicherheitsgrenze)
fig.add_shape(
    type="line",
    x0=0,
    x1=t_cycle,
    y0=30,
    y1=30,
    line=dict(color="red", width=2, dash="dash"),
    row=1,
    col=1,
)

# 3) Phasenmarkierungen (Inspiration / Exspiration) mit gestrichelten Linien & vertikaler Beschriftung
# Trennlinie bei Ende
