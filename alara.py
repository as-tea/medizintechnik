import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

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
st.sidebar.header("Applet-Steuerung (ALARA)")

# 1. Basis-Dosisrate der Strahlenquelle
basis_dosisrate = st.sidebar.slider(
    "Basis-Dosisrate der Quelle (mSv/h)",
    min_value=1.0,
    max_value=50.0,
    value=10.0,
    step=1.0,
)

# 2. Aufenthaltsdauer (Time)
zeit = st.sidebar.slider(
    "Aufenthaltsdauer (t in Stunden)",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1,
)

# 3. Abstand (Distance)
abstand = st.sidebar.slider(
    "Abstand zur Quelle (r in Meter)",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.1,
)

# 4. Abschirmung (Shielding)
st.sidebar.subheader("Abschirmung & Materialien")
schutzkleidung = st.sidebar.checkbox("Bleischürze (0.35 mm Pb) anlegen")
schicht_blei_mm = st.sidebar.slider(
    "Zusätzliche Bleiwand-Dicke (mm)",
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

# --- SESSION STATE FÜR KURVENVERGLEICH ---
r_werte = np.linspace(0.5, 5.0, 50)
aktuelle_dosis_werte = [
    basis_dosisrate * zeit * (1.0 / (r**2)) * faktor_schuerze * faktor_wand
    for r in r_werte
]

current_params = (basis_dosisrate, zeit, schutzkleidung, schicht_blei_mm)

if "history_curve" not in st.session_state:
    st.session_state.history_curve = None
    st.session_state.old_params = current_params

if st.session_state.old_params != current_params:
    # Speichere die vorherige Kurve, bevor die Parameter überschrieben werden
    # (Wir nutzen hier den Zustand der vorangegangenen Berechnung als Historie)
    st.session_state.history_curve = st.session_state.get("current_curve",uelle_dosis_werte if 'uelle_dosis_werte' in locals() else aktuelle_dosis_werte)
    st.session_state.old_params = current_params

st.session_state.current_curve = aktuelle_dosis_werte

# --- HAUPTBEREICH: ZENTRIERTE HIGHLIGHT-ANZEIGE ---
st.markdown("---")

_, _, col_center, _, _ = st.columns([1, 1, 2, 1, 1])

with col_center:
    if gesamtdosis < 1.0:
        st.markdown(f"""
            <div style="
                background-color: #e8f5e9; 
                border: 2px solid #4caf50; 
                padding: 15px; 
                border-radius: 10px; 
                text-align: center; 
                color: #2e7d32;">
                <h3 style="margin: 0; color: #2e7d32;">📊 Resultierende Dosis</h3>
                <h1 style="margin: 10px 0; color: #2e7d32;"><b>{gesamtdosis:.1f} mSv</b></h1>
                <p style="margin: 0; font-style: italic;">Geringe Exposition (Normalbereich)</p>
            </div>
        """, unsafe_allow_html=True)
    elif gesamtdosis < 6.0:
        st.markdown(f"""
            <div style="
                background-color: #fffde7; 
                border: 2px solid #fbc02d; 
                padding: 15px; 
                border-radius: 10px; 
                text-align: center; 
                color: #f57f17;">
                <h3 style="margin: 0; color: #f57f17;">📊 Resultierende Dosis</h3>
                <h1 style="margin: 10px 0; color: #f57f17;"><b>{gesamtdosis:.1f} mSv</b></h1>
                <p style="margin: 0; font-style: italic;">Erhöhte Exposition (Überwachungsbereich)</p>
            </div>
        """, unsafe_allow_html=True)
    elif gesamtdosis < 20.0:
        st.markdown(f"""
            <div style="
                background-color: #fff3e0; 
                border: 2px solid #ff9800; 
                padding: 15px; 
                border-radius: 10px; 
                text-align: center; 
                color: #e65100;">
                <h3 style="margin: 0; color: #e65100;">📊 Resultierende Dosis</h3>
                <h1 style="margin: 10px 0; color: #e65100;"><b>{gesamtdosis:.1f} mSv</b></h1>
                <p style="margin: 0; font-style: italic;">Erhöhte Exposition (Kontrollbereich)</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="
                background-color: #ffebee; 
                border: 2px solid #f44336; 
                padding: 15px; 
                border-radius: 10px; 
                text-align: center; 
                color: #c62828;">
                <h3 style="margin: 0; color: #c62828;">📊 Resultierende Dosis</h3>
                <h1 style="margin: 10px 0; color: #c62828;"><b>{gesamtdosis:.1f} mSv</b></h1>
                <p style="margin: 0; font-style: italic;">Kritische Dosis! Grenzwert überschritten!</p>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---")

st.subheader("💡 Einfluss der 3 ALARA-Säulen in Ihrem Szenario:")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("### ⏱️ Time (Zeit)")
    st.write(
        f"Die Dosis wächst **linear** mit der Zeit. Bei doppelter Dauer verdoppelt sich Ihre Dosis."
    )

with col_b:
    st.markdown("### 📏 Distance (Abstand)")
    st.write(
        "Das **Abstandsgesetz ($I \propto 1/r^2$)** führt zu einer großen Reduktion bei vergleichsweise kleiner Änderung."
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
st.subheader("📈 Veranschaulichung des Abstandsgesetzes ($1/r^2$) im Vergleich")

# Matplotlib Diagramm für feste Y-Achse und Mehrfachkurven
fig, ax = plt.subplots(figsize=(10, 4))

# Vorherige Kurve plotten (falls vorhanden)
if st.session_state.history_curve is not None:
    ax.plot(
        r_werte, 
        st.session_state.history_curve, 
        label="Zuletzt gewählte Kurve", 
        color="#b0bec5", 
        linestyle="--", 
        linewidth=2
    )

# Aktuelle Kurve plotten
ax.plot(
    r_werte, 
    aktuelle_dosis_werte, 
    label="Aktuelle Kurve", 
    color="#1976d2", 
    linewidth=2.5
)

# Markierung des aktuellen gewählten Punktes
ax.scatter([abstand], [gesamtdosis], color="#d32f2f", zorder=5, label=f"Aktueller Punkt ({abstand}m, {gesamtdosis:.1f}mSv)")

ax.set_xlabel("Abstand (m)")
ax.set_ylabel("Gesamtdosis (mSv)")

# Fixierte und stabile Y-Achsenbegrenzung (verhindert unruhiges Springen beim Verschieben des Abstands)
max_y_grenze = max(basis_dosisrate * zeit * 4.0 * 1.1, 10.0)
ax.set_ylim(0, max_y_grenze)

ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper right")

st.pyplot(fig)
