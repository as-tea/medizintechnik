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
    st.
