import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Seitengestaltung
st.set_page_config(
    page_title="Lernpfad Laser-Gewebe-Interaktion", page_icon="🔬", layout="wide"
)

st.title("Laser-Gewebe-Interaktion in der Medizin")
st.markdown("""
Erkunden Sie mit diesem Applet durch manuelles Einstellen 
von Wellenlänge, Pulsdauer und Leistungsdichte, wie physikalische Parameter die Gewebe-Wechselwirkung und das primäre Absorptionsverhalten bestimmen!
""")

# --- SIDEBAR FÜR DIE MANUELLEN EINGABEPARAMETER ---
st.sidebar.header("🎛️ Laser-Parameter-Steuerung")

# 1. Wellenlänge (Wellenlängenbereich von UV 190 nm bis IR 10600 nm)
wellenlaenge = st.sidebar.slider(
    "Wellenlänge (nm)",
    min_value=190.0,
    max_value=10600.0,
    value=10600.0,
    step=10.0,
    help="Bestimmt die Farbe bzw. den Spektralbereich (UV, Sichtbar, NIR, IR)."
)

# 2. Expositionszeit / Pulsdauer (Logarithmische Skala oder feine Stufen für den Pfad)
log_zeit = st.sidebar.slider(
    "Expositionszeit (log10 Sekunden)",
    min_value=-12.0,
    max_value=3.0,
    value=-3.0,
    step=0.5,
    help="Beispiel: log(10^-3) = -3 = 1 ms (thermisch), -12 = 1 ps (disruptiv/Plasma)."
)
pulsdauer_s = 10**log_zeit

# 3. Leistungsdichte
leistungsdichte_log = st.sidebar.slider(
    "Leistungsdichte (log10 W/cm²)",
    min_value=-3.0,
    max_value=15.0,
    value=4.0,
    step=0.5,
    help="Gibt an, wie viel Leistung auf eine bestimmte Fläche konzentriert ist (Spotgröße)."
)
leistungsdichte = 10**leistungsdichte_log

# --- LERNPFAD-AUFGABEN IN DER SIDEBAR (Kombiniert mit Leistungsdichte als Fokus-Hebel) ---
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Interaktive Arbeitsaufgaben")
st.sidebar.markdown("""
Arbeiten Sie nacheinander folgende Schritte ab und notieren Sie Ihre Beobachtungen analog:

- **1. Das präzise Skalpell:**  
  Stellen Sie $\\lambda = 10600\\text{ nm}$ ein (Infrarot). Welches Chromophor dominiert? Wie tief dringt der Strahl ein?
- **2. Tiefe Gewebekoagulation:**  
  Wechseln Sie ins „optische Fenster“ (ca. $800 - 1100\\text{ nm}$, z. B. $1064\\text{ nm}$). Was ändert sich an der Eindringtiefe?
- **3. Gefäßselektivität:**  
  Wählen Sie den grün/blauen Bereich ($510 - 532\\text{ nm}$). Welches Zielgewebe absorbiert hier stark?
- **4. Der Fokus-Hebel (Leistungsdichte):**  
  Variieren Sie im thermischen Bereich gezielt die *Leistungsdichte*. Beobachten Sie im Diagramm, wie der Effekt von mürbem Erwärmen zum schnellen Abtrag (Vaporisation) umschlägt!
- **5. Bindungsbruch ohne Hitze:**  
  Stellen Sie den UV-Bereich ($193\\text{ nm}$) bei extrem kurzen Pulsen ein. Welcher Mechanismus greift hier?
""")


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

# Bestimmung des Wechselwirkungs-Regimes (basierend auf Diagramm Folie 47)
if pulsdauer_s >= 1.0 and leistungsdichte < 1e3:
    regime = "Photochemisch (z.B. Photodynamische Therapie)"
    box_color = "#e8f5e9" # grün
    border_color = "#4caf50"
    text_color = "#2e7d32"
elif 1e-6 <= pulsdauer_s <= 10.0 and 1e1 <= leistungsdichte <= 1e9:
    regime = "Photothermisch (Koagulation, Vaporisation, Karbonisation)"
    box_color = "#fff3e0" # orange
    border_color = "#ff9800"
    text_color = "#e65100"
elif pulsdauer_s <= 1e-8 and 1e8 <= leistungsdichte <= 1e12:
    regime = "Photoablation (Molekülbindungsbruch ohne thermische Schädigung)"
    box_color = "#e3f2fd" # blau
    border_color = "#2196f3"
    text_color = "#0d47a1"
elif pulsdauer_s <= 1e-11 and leistungsdichte > 1e12:
    regime = "Photodisruption / Plasma (Stoßwelle und Kavitationsblase)"
    box_color = "#f3e5f5" # lila
    border_color = "#ab47bc"
    text_color = "#4a148c"
else:
    regime = "Thermischer Übergangsbereich / Hohe Energiedichte"
    box_color = "#fffde7" # gelb
    border_color = "#fbc02d"
    text_color = "#f57f17"


# --- HAUPTBEREICH: METRIK-BOX (ZENTRIERT) ---
st.markdown("---")
_, _, col_center, _, _ = st.columns([1, 1, 2, 1, 1])

with col_center:
    st.markdown(f"""
        <div style="
            background-color: {box_color}; 
            border: 2px solid {border_color}; 
            padding: 15px; 
            border-radius: 10px; 
            text-align: center; 
            color: {text_color};">
            <h3 style="margin: 0; color: {text_color};">🔬 Live-Analyse der Wechselwirkung</h3>
            <p style="margin: 8px 0 4px 0; font-size: 1.1em;"><b>Wellenlänge:</b> {wellenlaenge:.0f} nm | <b>Puls:</b> {pulsdauer_s:.1e} s | <b>Leistungsdichte:</b> 10^{leistungsdichte_log:.1f} W/cm²</p>
            <h2 style="margin: 5px 0; color: {text_color}; font-size: 1.3em;">{regime}</h2>
            <hr style="border: 0; border-top: 1px solid {border_color}; margin: 8px 0;">
            <p style="margin: 2px 0;"><b>Dominantes Chromophor:</b> {chromophor}</p>
            <p style="margin: 2px 0;"><b>Optische Eindringtiefe:</b> {eindringtiefe}</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- HAUPTBEREICH: DIAGRAMM (Leistungsdichte vs. Expositionszeit analog zu Folie 47) ---
st.subheader("Klassifikation der Wechselwirkungsmechanismen")
st.write("Der rote Punkt zeigt Ihre aktuell gewählte Kombination aus Expositionszeit und Leistungsdichte im Diagramm:")

fig, ax = plt.subplots(figsize=(10, 4.5))

ax.set_xlim(-15, 3)
ax.set_ylim(-3, 16)

# Zonen skizzieren
ax.text(-10, 13, "Photodisruption", color="#4a148c", fontweight="bold", fontsize=9, ha="center")
ax.text(-10, 11, "Plasma-induzierte Ablation", color="#b71c1c", fontweight="bold", fontsize=9, ha="center")
ax.text(-6, 7, "Photoablation", color="#0d47a1", fontweight="bold", fontsize=9, ha="center")
ax.text(-1, 2, "Thermal (Koagulation/Vaporisation)", color="#e65100", fontweight="bold", fontsize=9, ha="center")
ax.text(1, -1, "Photochemisch (PDT)", color="#2e7d32", fontweight="bold", fontsize=9, ha="center")

# Aktueller Punkt der Studierenden
ax.scatter([log_zeit], [leistungsdichte_log], color="#d32f2f", s=150, zorder=5, marker="o", label=f"Aktuelle Einstellung")

ax.set_xlabel("Expositionszeit log10(s)")
ax.set_ylabel("Leistungsdichte log10(W/cm²)")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper right")
ax.set_title("Wirkungs-Matrix: Leistungsdichte über Expositionszeit")

st.pyplot(fig)
