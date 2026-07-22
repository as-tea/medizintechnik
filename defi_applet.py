import streamlit as st
import numpy as np
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DefiSim – Defibrillation verstehen",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS für Schönheitsanpassungen
st.markdown("""
    <style>
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .info-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #E63946;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HEADER & EINLEITUNG
# -----------------------------------------------------------------------------
st.title("⚡ DefiSim: Physik & Physiologie der Defibrillation")
st.markdown("""
Dieses interaktive Applet erklärt die physikalischen und physiologischen Grundlagen der kardialen Defibrillation.
Ein Defibrillator stoppt durch eine gleichzeitige Depolarisation aller Herzmuskelzellen das "chaotische" Kammerflimmern. Durch diesen "Reset" kann der **Sinusknoten** seine Aufgabe als primärer Taktgeber wieder übernehmen.
""")

st.divider()

# -----------------------------------------------------------------------------
# SIDEBAR (PARAMETER-EINSTELLUNGEN)
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Parameter-Einstellungen")

energy = st.sidebar.slider(
    "Energie (Joule)",
    min_value=50, max_value=360, value=200, step=10,
    help="Die im Kondensator gespeicherte Energie E = 0.5 * C * U²"
)

impedance = st.sidebar.slider(
    "Impedanz / Widerstand (Ohm)",
    min_value=30, max_value=180, value=75, step=5,
    help="Der elektrische Widerstand des Patienten (Haut, Fettgewebe, Thorax)."
)

show_mds = st.sidebar.checkbox("Monophasisch gedämpfte Sinuskurve (MDS)", value=True)
show_bte = st.sidebar.checkbox("Biphasische Exponentialkurve (BTE)", value=True)
show_rbw = st.sidebar.checkbox("Biphasischer Rechteckimpuls (RBW)", value=True)

st.sidebar.markdown("---")
st.sidebar.info("**Tipp:** Verändern Sie die Impedanz, um zu sehen, wie sich die Kurvenformen bei unterschiedlichen Patientenwiderständen verschieben!")

# -----------------------------------------------------------------------------
# MATHEMATISCHE IMPULSFUNKTIONEN (Start bei t = 1 ms)
# -----------------------------------------------------------------------------
t = np.linspace(0, 20, 1000)  # Zeitstrahl in Millisekunden (0 bis 20 ms)

def calc_mds(E, R, t):
    """Monophasisch gedämpfte Sinuskurve (Start bei 1 ms)"""
    current = np.zeros_like(t)
    active_mask = t >= 1
    t_shift = t[active_mask] - 1
    
    i_peak = np.sqrt(2 * E / (0.003 * R)) * 0.4
    omega = 0.4
    alpha = 0.2 + (R / 200)
    
    val = i_peak * np.sin(omega * t_shift) * np.exp(-alpha * t_shift)
    current[active_mask] = np.where(val < 0, 0, val)
    return current

def calc_bte(E, R, t):
    """Biphasisch abgeschnittene Exponentialkurve (Start bei 1 ms)"""
    i_peak1 = np.sqrt(2 * E / 0.002) / (R * 0.5)
    tau = (R * 120) / 1000  # Zeitkonstante
    
    phase1_mask = (t >= 1) & (t <= 7)
    phase2_mask = (t > 7) & (t <= 11)
    
    current = np.zeros_like(t)
    # Phase 1 (Positiv)
    current[phase1_mask] = i_peak1 * np.exp(-(t[phase1_mask] - 1) / tau)
    # Phase 2 (Negativ / Invertiert)
    i_end_p1 = i_peak1 * np.exp(-6 / tau)
    current[phase2_mask] = -0.5 * i_end_p1 * np.exp(-(t[phase2_mask] - 7) / tau)
    return current

def calc_rbw(E, R, t):
    """Biphasischer Rechteckimpuls (Start bei 1 ms)"""
    i_const = np.sqrt(E / 12) * (100 / (R + 30)) * 3.5
    
    phase1_mask = (t >= 1) & (t <= 7)
    phase2_mask = (t > 7.2) & (t <= 12)
    
    current = np.zeros_like(t)
    current[phase1_mask] = i_const
    current[phase2_mask] = -0.6 * i_const
    return current

# -----------------------------------------------------------------------------
# HAUPTFUNKTION 1: WELLENFORM-GENERATOR
# -----------------------------------------------------------------------------
st.header("1. Impulsformen im Vergleich")

col_plot, col_stats = st.columns([3, 1])

fig = go.Figure()

i_mds = calc_mds(energy, impedance, t)
i_bte = calc_bte(energy, impedance, t)
i_rbw = calc_rbw(energy, impedance, t)

if show_mds:
    fig.add_trace(go.Scatter(x=t, y=i_mds, mode='lines', name='MDS (Monophasisch)', line=dict(color='#D90429', width=3)))
if show_bte:
    fig.add_trace(go.Scatter(x=t, y=i_bte, mode='lines', name='BTE (Biphasisch Exponentiell)', line=dict(color='#023E8A', width=3)))
if show_rbw:
    fig.add_trace(go.Scatter(x=t, y=i_rbw, mode='lines', name='RBW (Biphasisch Rechteck)', line=dict(color='#00B4D8', width=3, dash='dash')))

fig.update_layout(
    title=f"Stromverlauf über die Zeit (bei {energy} J und {impedance} Ω)",
    xaxis_title="Zeit (ms)",
    yaxis_title="Stromstärke I (Ampere)",
    template="plotly_white",
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    height=450
)

# Fixierte Achsenskalierung zur Vermeidung automatischer Neuskalierung
fig.update_xaxes(range=[0, 20], autorange=False)
fig.update_yaxes(range=[-35, 65], autorange=False)

with col_plot:
    st.plotly_chart(fig, use_container_width=True)

with col_stats:
    st.subheader("Kennzahlen")
    peak_mds = np.max(i_mds) if show_mds else 0
    peak_bte = np.max(i_bte) if show_bte else 0
    peak_rbw = np.max(i_rbw) if show_rbw else 0
    
    st.metric(label="Peak Current (MDS)", value=f"{peak_mds:.1f} A" if show_mds else "N/A")
    st.metric(label="Peak Current (BTE)", value=f"{peak_bte:.1f} A" if show_bte else "N/A")
    st.metric(label="Peak Current (RBW)", value=f"{peak_rbw:.1f} A" if show_rbw else "N/A")

# -----------------------------------------------------------------------------
# INFOBOX: BTE VS. RBW
# -----------------------------------------------------------------------------
st.subheader("🔍 Detailvergleich: BTE vs. RBW")

col_bte_info, col_rbw_info = st.columns(2)

with col_bte_info:
    st.markdown("""
    <div class="info-card">
        <h4>Biphasisch Exponentiell (BTE)</h4>
        <ul>
            <li><b>Funktionsweise:</b> Der Kondensator entlädt sich exponentiell. Nach einigen Millisekunden wird der Strom abgebrochen und umgekehrt.</li>
            <li><b>Impedanz-Verhalten:</b> Bei hoher Impedanz fällt die Kurve langsamer ab. Die Impulsdauer wird durch komplexe Steuerlogik abhängig von der Impedanz automatisch angepasst.</li>
            <li><b>Fazit:</b> Sehr gut erforscht, weltweiter Standard in vielen AEDs.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_rbw_info:
    st.markdown("""
    <div class="info-card">
        <h4>Biphasischer Rechteckimpuls (RBW)</h4>
        <ul>
            <li><b>Funktionsweise:</b> Hält den Stromfluss in Phase 1 konstant auf einem Plateau (kein exponentieller Abfall).</li>
            <li><b>Impedanz-Verhalten:</b> Der Defibrillator passt die Spannung aktiv an, um stets die ideale Stromstärke zu liefern.</li>
            <li><b>Fazit:</b> Vermeidet unnötig hohe Stromspitzen. Besonders effektiv bei Patienten mit hoher Impedanz (z. B. adipös). ABER: Komplexe, teure Elektronik.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# HAUPTFUNKTION 2: ELEKTRONENFLUSS & HERZ-SIMULATION
# -----------------------------------------------------------------------------
st.header("2. Warum ist biphasische Stimulation 'gesünder'?")

mode = st.radio("Wählen Sie den Impulstyp zur Demonstration des Elektronenflusses:", 
                ["Monophasisch (MDS)", "Biphasisch (BTE / RBW)"], horizontal=True)

col_heart_vis, col_heart_text = st.columns([1, 1])

def generate_heart_diagram(mode):
    fig = go.Figure()
    
    # Herz-Kontur (Schematisch)
    fig.add_shape(type="path",
        path="M 0,0.5 C -0.8,1.5 -1.8,0.5 -1.2,-0.5 L 0,-1.8 L 1.2,-0.5 C 1.8,0.5 0.8,1.5 0,0.5 Z",
        fillcolor="#ffccd5", line_color="#c9184a", line_width=3
    )
    
    # Elektroden (Anode & Kathode)
    fig.add_shape(type="rect", x0=-1.8, y0=0.8, x1=-1.2, y1=1.4, fillcolor="#3a86ff", line_color="white", name="Anode")
    fig.add_annotation(x=-1.5, y=1.1, text="Anode", showarrow=False, font=dict(color="white", size=12))
    
    fig.add_shape(type="rect", x0=1.2, y0=-1.4, x1=1.8, y1=-0.8, fillcolor="#3a86ff", line_color="white", name="Kathode")
    fig.add_annotation(x=1.5, y=-1.1, text="Kathode", showarrow=False, font=dict(color="white", size=12))

    if mode == "Monophasisch (MDS)":
        # Nur eine Richtung (Anode -> Kathode) mit Beschriftung am Pfeil
        fig.add_annotation(x=0.8, y=-0.4, ax=-0.8, ay=0.4, xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=2, arrowsize=2, arrowwidth=4, arrowcolor="#023e8a",
                            text="Stromfluss", font=dict(color="#023e8a", size=12), yshift=15)

    else:
        # Phase 1 (Anode -> Kathode)
        fig.add_annotation(x=0.6, y=-0.2, ax=-0.6, ay=0.4, xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=3, arrowcolor="#023e8a",
                            text="Stromfluss (Phase 1)", font=dict(color="#023e8a", size=11), yshift=15)
        # Phase 2 (Kathode -> Anode)
        fig.add_annotation(x=-0.4, y=-0.6, ax=0.4, ay=-1.0, xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=3, arrowcolor="#00b4d8",
                            text="Stromfluss (Phase 2)", font=dict(color="#00b4d8", size=11), yshift=-15)

    fig.update_xaxes(visible=False, range=[-2.5, 2.5])
    fig.update_yaxes(visible=False, range=[-2.2, 2.0])
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white")
    return fig

with col_heart_vis:
    st.plotly_chart(generate_heart_diagram(mode), use_container_width=True)

with col_heart_text:
    if mode == "Monophasisch (MDS)":
        st.error("### Nachteile der monophasischen Stimulation")
        st.markdown("""
        * **Einheitlicher Stromfluss:** Der Strom fließt ausschließlich von der Anode zur Kathode.
        * **Hohe Stromspitzen erforderlich:** Um tief liegende Herzmuskelschichten zu erreichen, muss mit sehr hoher Energie (bis zu **360 Joule**) gearbeitet werden.
        * **Gewebebelastung:** Der hohe Spitzenstrom schädigt das Myokard und führt zu thermischen Belastungen an den Kontaktstellen.
        """)
    else:
        st.success("### Vorteile der biphasischen Stimulation")
        st.markdown("""
        * **Phase 1 (Depolarisation):** Der Strom fließt von der Anode zur Kathode und depolarisiert den Großteil des Herzmuskels.
        * **Phase 2 (Repolarisation/Korrektur):** Die Stromrichtung kehrt sich um. Dies stellt das elektrische Potential an den Zellmembranen wieder her und entfernt verbliebene Ladungsüberschüsse.
        * **Geringere Energie nötig:** Bereits **150 bis 200 Joule** reichen aus.
        * **Zellschonend:** Signifikant geringeres Risiko für Verbrennungen und myokardiale Schädigung nach der Defibrillation.
        """)
