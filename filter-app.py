import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, filtfilt

# --- EKG-Signal Generierungsfunktion ---
@st.cache_data
def generiere_ekg_signal(Fs_Abtastrate, Zeit_Sekunden, Amplitude_Netz, Amplitude_Rauschen):
    t_Vektor = np.arange(0, Zeit_Sekunden, 1/Fs_Abtastrate)
    
    # 1. Sauberes EKG (Approximation)
    Herzrate_BPM = 75
    T_Schlag = 60 / Herzrate_BPM
    ekg_sauber = np.zeros_like(t_Vektor)
    
    for k in range(int(Zeit_Sekunden / T_Schlag)):
        t_schlag = t_Vektor - k * T_Schlag
        R_Welle = 1.0 * np.exp(-((t_schlag - 0.3)**2) / (2 * 0.02**2))
        T_Welle = 0.4 * np.exp(-((t_schlag - 0.6)**2) / (2 * 0.08**2))
        P_Welle = 0.2 * np.exp(-((t_schlag - 0.1)**2) / (2 * 0.05**2))
        ekg_sauber += R_Welle + T_Welle + P_Welle
        
    # 2. Störungen hinzufügen
    Rauschen_Breitband = Amplitude_Rauschen * np.random.randn(len(t_Vektor))
    
    # Basislinien-Drift (langsames Artefakt)
    Drift_Sinus = 0.6 * np.sin(2 * np.pi * 0.2 * t_Vektor) 
    
    # Netzbrummen (50 Hz)
    Frequenz_Netzbrummen = 50 
    Rauschen_Netz = Amplitude_Netz * np.sin(2 * np.pi * Frequenz_Netzbrummen * t_Vektor)

    EKG_gestoert = ekg_sauber + Rauschen_Breitband + Rauschen_Netz + Drift_Sinus
    return t_Vektor, EKG_gestoert, Fs_Abtastrate

# --- Filter-Funktionsblock ---
def wende_filter_an(ekg_signal, Fs_Abtastrate, tp_aktiv, F_Grenz_TP, hp_aktiv, F_Grenz_HP, kerb_aktiv, F_Kerb, BW_Kerb):
    
    # Start mit dem gestörten Signal
    ekg_gefiltert = ekg_signal.copy()
    
    # 1. Tiefpassfilter (Hochfrequenzrauschen)
    if tp_aktiv:
        F_norm = F_Grenz_TP / (Fs_Abtastrate / 2)
        b, a = butter(4, F_norm, 'low')
        ekg_gefiltert = filtfilt(b, a, ekg_gefiltert)
        
    # 2. Hochpassfilter (Basislinien-Drift)
    if hp_aktiv:
        F_norm = F_Grenz_HP / (Fs_Abtastrate / 2)
        b, a = butter(4, F_norm, 'high')
        ekg_gefiltert = filtfilt(b, a, ekg_gefiltert)
        
    # 3. Kerbfilter (50 Hz Netzbrummen)
    if kerb_aktiv:
        Wo_norm = F_Kerb / (Fs_Abtastrate / 2)
        BW_norm = BW_Kerb / (Fs_Abtastrate / 2)
        b, a = iirnotch(Wo_norm, BW_norm)
        ekg_gefiltert = filtfilt(b, a, ekg_gefiltert)
        
    return ekg_gefiltert

# --- Streamlit UI-Design ---
st.set_page_config(layout="wide", page_title="EKG Filter App")
st.title("EKG Filter Simulation: Entfernen von Artefakten 🩺")

# --- Seitenleiste für Rauschen & Generierung ---
st.sidebar.header("Signal-Konfiguration (Artefakte) ⚙️")

amp_netz = st.sidebar.slider("Amplitude 50 Hz Netzbrummen", 0.0, 1.0, 0.5, 0.1)
amp_rauschen = st.sidebar.slider("Amplitude Breitbandrauschen", 0.0, 0.5, 0.2, 0.05)
Fs = st.sidebar.slider("Abtastrate (Fs)", 100, 500, 250, 50)
zeit = 10 # Feste Signaldauer

# Generiere das Signal
t, ekg_original, Fs_rate = generiere_ekg_signal(Fs, zeit, amp_netz, amp_rauschen)

# --- Hauptbereich für Filtersteuerung ---
st.header("Filter-Einstellungen")

col1, col2, col3 = st.columns(3)

# Tiefpass-Block
with col1:
    st.subheader("1. Tiefpassfilter")
    tp_aktiv = st.checkbox("Tiefpassfilter aktivieren (Entfernt Hochfrequenzrauschen)", value=True)
    F_Grenz_TP = st.slider("Grenzfrequenz (Hz)", 5, 100, 40, 5, disabled=not tp_aktiv)
    st.caption("EKG-Standard: ca. 30-40 Hz. Zu tief filtert QRS-Spitze weg.")

# Hochpass-Block
with col2:
    st.subheader("2. Hochpassfilter")
    hp_aktiv = st.checkbox("Hochpassfilter aktivieren (Entfernt Basislinien-Drift)", value=True)
    F_Grenz_HP = st.slider("Grenzfrequenz (Hz)", 0.01, 2.0, 0.5, 0.05, disabled=not hp_aktiv)
    st.caption("EKG-Standard: ca. 0.05 - 1 Hz. Zu hoch verzerrt ST-Strecke.")
    
# Kerbfilter-Block
with col3:
    st.subheader("3. Kerbfilter (Notch)")
    kerb_aktiv = st.checkbox("Kerbfilter aktivieren (Entfernt 50 Hz Netzbrummen)", value=True)
    F_Kerb = st.slider("Mittenfrequenz (Hz)", 40, 60, 50, 1, disabled=not kerb_aktiv)
    BW_Kerb = st.slider("Bandbreite (Hz)", 0.5, 5.0, 1.0, 0.5, disabled=not kerb_aktiv)
    st.caption("Eliminiert eine einzige Frequenz (50 Hz).")

# --- Filterung anwenden ---
ekg_gefiltert = wende_filter_an(ekg_original, Fs_rate, tp_aktiv, F_Grenz_TP, hp_aktiv, F_Grenz_HP, kerb_aktiv, F_Kerb, BW_Kerb)

# --- Plot-Bereich ---
st.header("Visualisierung")

fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Plot 1: Gestörtes Signal (Original)
ax[0].plot(t, ekg_original, label='Gestörtes Signal', color='red', linewidth=0.8)
ax[0].set_title('Ursprüngliches, gestörtes EKG-Signal')
ax[0].set_ylabel('Amplitude (mV)')
ax[0].grid(True)
ax[0].legend()

# Plot 2: Gefiltertes Signal
ax[1].plot(t, ekg_gefiltert, label='Gefiltertes Signal', color='blue', linewidth=1.5)
ax[1].set_title('Gefiltertes EKG-Signal')
ax[1].set_xlabel('Zeit (s)')
ax[1].set_ylabel('Amplitude (mV)')
ax[1].grid(True)
ax[1].legend()

# Anzeige im Streamlit
st.pyplot(fig)
