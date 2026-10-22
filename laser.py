import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Seitengestaltung
st.set_page_config(
    page_title="Lernpfad Laser-Gewebe-Interaktion", page_icon="🔬", layout="wide"
)

st.title("Laser-Gewebe-Interaktion in der Medizin")
st.markdown("""
Erkunden Sie mit diesem Applet durch manuelles Einstellen 
von Wellenlänge, Pulsdauer und Leistungsdichte, wie physikalische Parameter die Gewebe-Wechselwirkung und das primäre Absorptionsverhalten bestimmen!
""")

# --- 1. ARBEITSAUFTRÄGE ALS AUFKLAPPBARER BEREICH ÜBER DEM ANZEIGEFENSTER ---
with st.expander("📋 Interaktive Arbeitsaufgaben (Lernpfad einblenden)", expanded=True):
    st.markdown("""
    Arbeiten Sie nacheinander folgende Schritte ab und notieren Sie Ihre Beobachtungen analog:
    - **1. Das präzise Skalpell:** Stellen Sie $\\lambda = 10600\\text{ nm}$ ein (Infrarot). Welches Chromophor dominiert? Wie tief dringt der Strahl ein?
    - **2. Tiefe Gewebekoagulation:** Wechseln Sie ins „optische Fenster“ (ca. $800 - 1100\\text{ nm}$, z. B. $1064\\text{ nm}$). Was ändert sich an der Eindringtiefe? Wieso?
    - **3. Gefäßselektivität:** Wählen Sie den grün/blauen Bereich ($510 - 532\\text{ nm}$). Welches Zielgewebe absorbiert hier stark?
    - **4. Der Fokus-Hebel (Leistungsdichte):** Variieren Sie im thermischen Bereich gezielt die Leistungsdichte. Beobachten Sie im Diagramm, wie sich die Wechselwirkung ändert.
    - **5. Bindungsbruch ohne Hitze:** Stellen Sie den UV-Bereich ($193\\text{ nm}$) bei extrem kurzen Pulsen ein. Welcher Mechanismus greift hier?
    - **6. Laserarten:** Welchen Lasertypen entsprechen die Einstellungen aus den Aufgaben 1.-5.?
    """)

# --- SIDEBAR FÜR DIE MANUELLEN EINGABEPARAMETER ---
st.sidebar.header("🎛️ Laser-Parameter-Steuerung")

# Spektralbereich-Auswahl für lineare, hochauflösende Slider
bereich = st.sidebar.selectbox(
    "Spektralbereich wählen",
    [
        "UV & Sichtbar (190 - 780 nm)", 
        "Optisches Fenster / NIR (780 - 1200 nm)", 
        "Infrarot / CO2 (1200 - 10600 nm)"
    ],
    key="spektral_bereich"
)

if bereich == "UV & Sichtbar (190 - 780 nm)":
    min_v, max_v, step_v = 190.0, 780.0, 1.0
elif bereich == "Optisches Fenster / NIR (780 - 1200 nm)":
    min_v, max_v, step_v = 780.0, 1200.0, 2.0
else:
    min_v, max_v, step_v = 1200.0, 10600.0, 10.0

# Session State Logik, damit der Regler beim Bereichswechsel ganz links startet
if "current_bereich" not in st.session_state:
    st.session_state.current_bereich = bereich
    st.session_state.wellenlaenge_input = min_v

if st.session_state.current_bereich != bereich:
    st.session_state.current_bereich = bereich
    st.session_state.wellenlaenge_input = min_v

wellenlaenge = st.sidebar.slider(
    "Wellenlänge (nm) - Linear",
    min_value=min_v,
    max_value=max_v,
    step=step_v,
    key="wellenlaenge_input",
    help="Lineare Skala für den gewählten Bereich zur feinen Auflösung."
)

# Expositionszeit mit e-Formalismus
log_zeit = st.sidebar.slider(
    "Expositionszeit (log10 Sekunden)",
    min_value=-12.0,
    max_value=3.0,
    value=-3.0,
    step=0.5,
    help="Beispiel: -3 = 1.0e-3 s (thermisch), -12 = 1.0e-12 s (disruptiv/Plasma)."
)
pulsdauer_s = 10**log_zeit

# Leistungsdichte mit e-Formalismus
leistungsdichte_log = st.sidebar.slider(
    "Leistungsdichte (log10 W/cm²)",
    min_value=-3.0,
    max_value=15.0,
    value=4.0,
    step=0.5,
    help="Gibt an, wie viel Leistung auf eine bestimmte Fläche konzentriert ist (Spotgröße)."
)
leistungsdichte = 10**leistungsdichte_log


# --- LOGISCHE AUSWERTUNG DER PHYSIKALISCHEN EFFEKTE ---
if wellenlaenge < 300:
    chromophor = "Proteine & Nukleinsäuren (UV-Absorption)"
    eindringtiefe = "Extrem gering (< 0,01 mm)"
elif 400 <= wellenlaenge <= 600:
    chromophor = "Hämoglobin & Melanin (Starke Absorption)"
    eindringtiefe = "Gering bis mittel (~ 0,1 - 1 mm)"
elif 600 < wellenlaenge <= 1200:
    chromophor = "Melanin & schwach Wasser/Hämoglobin ('Optisches Fenster')"
    eindringtiefe = "Tief (mehrere mm bis cm)"
elif 1200 < wellenlaenge < 2500:
    chromophor = "Wasser (beginnende Absorption)"
    eindringtiefe = "Mittel (~ 1 mm)"
else: 
    chromophor = "Wasser (Starke Resonanz / Absorptionsbande)"
    eindringtiefe = "Sehr gering (< 0,1 mm, Oberflächenwirkung)"

# Harmonisierte Regime-Bedingungen passend zu den Diagramm-Kreisen
if log_zeit <= -9.5 and leistungsdichte_log >= 11.0:
    regime = "Photodisruption / Plasma (Stoßwelle und Kavitationsblase)"
elif -9.5 < log_zeit <= -8.5 and leistungsdichte_log >= 9.0:
    regime = "Plasma-induzierte Ablation"
elif -8.0 <= log_zeit <= -4.0 and 5.0 <= leistungsdichte_log <= 10.0:
    regime = "Photoablation (Molekülbindungsbruch ohne thermische Schädigung)"
elif log_zeit >= -4.0 and leistungsdichte_log <= 10.0:
    regime = "Photothermisch (Koagulation, Vaporisation, Karbonisation)"
elif log_zeit >= 0.0 and leistungsdichte_log < 3.0:
    regime = "Photochemisch (z.B. Photodynamische Therapie)"
else:
    regime = "Thermischer Übergangsbereich / Hohe Energiedichte"


# --- ANZEIGEFENSTER (Monochrom & e-Notation) ---
st.markdown("---")
_, _, col_center, _, _ = st.columns([1, 1, 2, 1, 1])

with col_center:
    st.markdown(f"""
        <div style="
            background-color: #f8f9fa; 
            border: 2px solid #6c757d; 
            padding: 15px; 
            border-radius: 10px; 
            text-align: center; 
            color: #212529;">
            <h3 style="margin: 0; color: #343a40;">🔬 Live-Analyse der Wechselwirkung</h3>
            <p style="margin: 8px 0 4px 0; font-size: 1.0em;"><b>Wellenlänge:</b> {wellenlaenge:.1f} nm | <b>Puls:</b> {pulsdauer_s:.1e} s | <b>Leistungsdichte:</b> 1.0e{leistungsdichte_log:.1f} W/cm²</p>
            <h2 style="margin: 5px 0; color: #0056b3; font-size: 1.2em;">{regime}</h2>
            <hr style="border: 0; border-top: 1px solid #ced4da; margin: 8px 0;">
            <p style="margin: 2px 0;"><b>Dominantes Chromophor:</b> {chromophor}</p>
            <p style="margin: 2px 0;"><b>Optische Eindringtiefe:</b> {eindringtiefe}</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- DIAGRAMM MIT HALBTRANSPARENTEN KREISEN ---
st.subheader("Klassifikation der Wechselwirkungsmechanismen")
st.write("Der rote Punkt zeigt Ihre aktuell gewählte Kombination aus Expositionszeit und Leistungsdichte im Diagramm:")

fig, ax = plt.subplots(figsize=(10, 5))

ax.set_xlim(-15, 3)
ax.set_ylim(-3, 16)

# Halbtransparente Kreise für die Wirkungsbereiche (analog zu Folie 47)
ax.add_patch(patches.Circle((-10.5, 13.5), radius=2.2, color='green', alpha=0.25, ec='none'))
ax.add_patch(patches.Circle((-11.0, 11.0), radius=2.2, color='salmon', alpha=0.25, ec='none'))
ax.add_patch(patches.Circle((-6.0, 7.0), radius=1.8, color='dodgerblue', alpha=0.25, ec='none'))
ax.add_patch(patches.Circle((-1.5, 3.5), radius=3.2, color='sandybrown', alpha=0.25, ec='none'))
ax.add_patch(patches.Circle((1.5, -0.5), radius=1.8, color='gold', alpha=0.25, ec='none'))

# Beschriftungen der Zonen
ax.text(-10.5, 13.5, "Photodisruption", color="#2e7d32", fontweight="bold", fontsize=9, ha="center", va="center")
ax.text(-11.0, 11.0, "Plasma-induced\nablation", color="#c62828", fontweight="bold", fontsize=9, ha="center", va="center")
ax.text(-6.0, 7.0, "Photoablation", color="#1565c0", fontweight="bold", fontsize=9, ha="center", va="center")
ax.text(-1.5, 3.5, "Thermal", color="#d84315", fontweight="bold", fontsize=10, ha="center", va="center")
ax.text(1.5, -0.5, "Photochemical", color="#f57f17", fontweight="bold", fontsize=9, ha="center", va="center")

# Aktueller Punkt der Studierenden
ax.scatter([log_zeit], [leistungsdichte_log], color="#d32f2f", s=180, zorder=5, marker="o", label=f"Aktuelle Einstellung")

ax.set_xlabel("Expositionszeit (s) [log10]")
ax.set_ylabel("Leistungsdichte (W/cm²) [log10]")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper right")
ax.set_title("Wirkungs-Matrix: Leistungsdichte über Expositionszeit (mit Regime-Zonen)")

st.pyplot(fig)
