import numpy as np
import plotly.graph_objects as go
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
    help="Niedrig = Steife Lunge, Hoch = Dehnbare Lunge",
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
        "Tidalvolumen $V_T$ (mL)", min_value=200, max_value=800, value=500, step=50
    )
    p_insp_target = None
else:
    p_insp_target = st.sidebar.slider(
        "Inspirationsdruck $p_insp$ (mbar)",
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
#        "Flow V'(t) [L/min]",
        "Volumen V(t) [mL]",
    ),
)

fig.add_trace(
    go.Scatter(x=t, y=p_t, name="Druck", line=dict(color="blue", width=2)),
    row=1,
    col=1,
)
#fig.add_trace(
#    go.Scatter(x=t, y=flow_t, name="Flow", line=dict(color="teal", width=2)),
#    row=2,
#    col=1,
)
fig.add_trace(
    go.Scatter(x=t, y=v_t, name="Volumen", line=dict(color="green", width=2)),
    row=3,
    col=1,
)

# Rote Linie bei 30 mbar (Sicherheitsgrenze)
fig.add_hline(
    y=30,
    line_width=2,
    line_dash="dash",
    line_color="red",
    row=1,
    col=1
)

# Phasenmarkierung (Inspiration / Exspiration)
fig.add_vline(
    x=t_insp,
    line_width=1,
    line_dash="dash",
    line_color="gray"
)

# Vertikale Beschriftung der Phasen
fig.add_annotation(
    x=t_insp / 2,
    y=50,
    text="INSPIRATION",
    showarrow=False,
    textangle=0,
    font=dict(size=11, color="gray"),
    row=1,
    col=1
)
fig.add_annotation(
    x=t_insp + (t_cycle - t_insp) / 2,
    y=50,
    text="EXSPIRATION",
    showarrow=False,
    textangle=0,
    font=dict(size=11, color="gray"),
    row=1,
    col=1
)

# Fixierte Skalierung der Achsen
fig.update_xaxes(range=[0, t_cycle], title_text="Zeit [s]", row=3, col=1)
fig.update_yaxes(range=[0, 70], row=1, col=1)      # Druckachse fest 0 bis 70 mbar
fig.update_yaxes(range=[-100, 100], row=2, col=1)  # Flowachse fest -100 bis +100 L/min
fig.update_yaxes(range=[0, 1000], row=3, col=1)    # Volumenachse fest 0 bis 1000 mL

fig.update_layout(height=650, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))

# --- DASHBOARD LAYOUT ---
col1, col2 = st.columns([2.5, 1])

with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Messwerte")

    max_p = np.max(p_t)
    max_v = np.max(v_t)

    st.metric("Spitzendruck ($p_peak$)", f"{max_p:.1f} mbar")
    st.metric("Erzieltes Tidalvolumen ($V_T$)", f"{max_v:.0f} mL")

    st.markdown("---")

    # Vergrößerte Statusmeldungen für Druck und Volumen
    st.subheader("Status & Warnungen")

    if max_p > 30:
        st.error("🚨 **BAROTRAUMA-RISIKO!**\n\nDer Spitzendruck liegt über 30 mbar!")
    else:
        st.success("✅ **DRUCK OPTIMAL**\n\nSpitzendruck im sicheren Bereich (≤ 30 mbar).")

    if max_v < 300:
        st.warning("⚠️ **HYPOVENTILATION!**\n\nTidalvolumen ist kritisch niedrig (< 300 mL).")
    elif max_v > 700:
        st.warning("⚠️ **VOLUTRAUMA-RISIKO!**\n\nTidalvolumen ist sehr hoch (> 700 mL).")
    else:
        st.info("ℹ️ **VOLUMEN NORMBEREICH**\n\nTidalvolumen liegt zwischen 300 und 700 mL.")

# --- ARBEITSAUFTRÄGE (Aufklappbare Box) ---
with st.expander("📋 **Arbeitsaufträge für Studierende (Hier klicken zum Aufklappen)**"):
    st.markdown("""
    **1. Veränderung der Compliance**
    * Wählen Sie zuerst den **VCV-Modus** ($V_T = 500\\text{ mL}$). Senken Sie die Compliance schrittweise von $60\\text{ mL/mbar}$ auf $20\\text{ mL/mbar}$. Beobachten Sie, was mit dem Spitzendruck ($p_{\\text{peak}}$) passiert.
    * Wechseln Sie nun in den **PCV-Modus** ($p_{\\text{insp}} = 20\\text{ mbar}$). Senken Sie die Compliance erneut auf $20\\text{ mL/mbar}$. Was passiert hier mit dem erzielten Atemzugvolumen ($V_T$)?

    **2. Das Risiko für zu hohen Druck (VCV):**
    * Finden Sie im VCV-Modus bei einer sehr steifen Lunge ($C = 20\\text{ mL/mbar}$) ein Tidalvolumen, bei dem die Sicherheitsgrenze von $30\\text{ mbar}$ gerade noch **nicht** überschritten wird.

    **3. Zielvolumen sichern (PCV):**
    * Versuchen Sie im PCV-Modus bei $C = 30\\text{ mL/mbar}$ ein Tidalvolumen von $500\\text{ mL}$ zu erreichen. Wie hoch muss der Inspirationsdruck eingestellt sein? Wird dabei die Druckgrenze überschritten?
    """)
