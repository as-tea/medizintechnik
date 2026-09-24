import numpy as np
import streamlit as st

# Seitengestaltung
st.set_page_config(
    page_title="ALARA-Strahlenschutz-Applet", page_icon="☢️", layout="wide"
)

st.title("☢️ ALARA-Strahlenschutz-Applet")
st.markdown("""
Dieses Applet demonstriert das **ALARA-Prinzip** (*As Low As Reasonably Achievable* – So viel wie nötig, so wenig wie möglich). 
Sie können die Parameter über die Schieberegler und Schalter verändern, um die resultierende Dosis live zu beobachten!
""")

# Sidebar für die Eingabeparameter
st.sidebar.header("🎛️ Applet-Steuerung (ALARA)")

# 1. Basis-Dosisrate der Strahlenquelle
basis_dosisrate = st.sidebar.slider(
    "Basis-Dosisrate der Quelle [mSv/h]",
    min_value=1.0,
    max_value=100.0,
    value=20.0,
    step=1.0,
)

# 2. Aufenthaltsdauer (Time)
zeit = st.sidebar.slider(
    "⏱️ Aufenthaltsdauer (t) [Stunden]",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1,
)

# 3. Abstand (Distance)
abstand = st.sidebar.slider(
    "📏 Abstand zur Quelle (r) [Meter]",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.1,
)

# 4. Abschirmung (Shielding)
st.sidebar.subheader("🛡️ Abschirmung & Materialien")
schutzkleidung = st.sidebar.checkbox("Bleischürze (0.35 mm Pb) anlegen")
schicht_blei_mm = st.sidebar.slider(
    "Zusätzliche Bleiwand-Dicke [mm]",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.5,
)

# Info-Box in der Sidebar
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ Übliche Dosisraten im Feld"):
    st.markdown("""
    * **Interventionell (C-Bogen):** ~1 bis 10 mSv/h (Streudosis ungeschützt)
    * **Nuklearmedizin (Heißlabor):** Oft mSv/h bis Dutzende mSv/h an Quellen
    * **Diagnostik (geschützt):** < 0,01 bis < 1 mSv/h
    * **Berufl. Jahresgrenzwert:** 20 mSv/Jahr
    """)

# --- BERECHNUNG DER DOSIS ---
faktor_abstand = 1.0 / (abstand**2)
faktor_schuerze = 0.3 if schutzkleidung else 1.0
mu_blei = 0.6  
faktor_wand = np.exp(-mu_blei * schicht_blei_mm)

dosis_stündlich = (
    basis_dosisrate * faktor_abstand * faktor_schuerze * faktor_wand
)
gesamtdosis = dosis_stündlich * zeit  

# --- HAUPTBEREICH: ANZEIGE ---
# Status-Text für das Delta und Farb-Logik festlegen
if gesamtdosis < 1.0:
    status_text = "Geringe Exposition (Normalbereich)"
    delta_color_val = "normal"
elif gesamtdosis < 20.0:
    status_text = "Erhöhte Exposition (Überwachungsbereich)"
    delta_color_val = "off"
else:
    status_text = "Kritische Dosis! Grenzwert überschritten!"
    delta_color_val = "inverse"

# Wir erstellen 3 Spalten, um die Metrik in der Mitte (col2) zu zentrieren
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.metric(
        label="📊 Resultierende Dosis",
        value=f"{gesamtdosis:.3f} mSv",
        delta=status_text,
        delta_color=delta_color_val,
    )

st.markdown("---")

st.subheader("💡 Einfluss der 3 ALARA-Säulen in Ihrem Szenario:")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("### ⏱️ Time (Zeit)")
    st.write(
        f"Die Dosis wächst **linear** mit der Zeit. Bei doppelter Dauer ({zeit*2:.1f}h) verdoppelt sich Ihre Dosis."
    )

with col_b:
    st.markdown("### 📏 Distance (Abstand)")
    st.write(
        "Das **Abstandsgesetz ($I \propto 1/r^2$)** greift stark. "
        f"Durch Ihren gewählten Abstand von {abstand}m wird die Strahlung um den Faktor `{faktor_abstand:.2f}` skaliert."
    )

with col_c:
    st.markdown("### 🛡️ Shielding (Schutz)")
    schütz_status = (
        "Aktiv (Schürze + Wand)"
        if (schutzkleidung or schicht_blei_mm > 0)
        else "Keine"
    )
    st.write(f"Aktiver Schutz: **{schütz_status}**")
    st.write(
        f"Resttransmission durch Ihre Schutzschicht: `{faktor_schuerze * faktor_wand * 100:.1f}%`"
    )

st.markdown("---")
st.subheader("📈 Veranschaulichung des Abstandsgesetzes ($1/r^2$)")
r_werte = np.linspace(0.5, 5.0, 50)
dosis_werte = [
    basis_dosisrate * zeit * (1.0 / (r**2)) * faktor_schuerze * faktor_wand
    for r in r_werte
]

st.line_chart(
    data={"Abstand [m]": r_werte, "Dosis [mSv]": dosis_werte}, x="Abstand [m]"
)
