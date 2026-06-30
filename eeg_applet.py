import streamlit as st
import numpy as np
import scipy.signal as signal
import plotly.graph_objects as go

def eeg_applet():
    st.header("🧠 Der Gehirnwellen-Synthesizer & Filter")
    
    # Didaktischer Begleittext am Anfang
    st.info(
        "**Arbeitsaufgabe für Studierende:**\n"
        "1. Wählen Sie den Zustand **'Höchste Konzentration (Gamma)'**.\n"
        "2. Schalten Sie das **50 Hz Netzbrummen** ein. Fällt Ihnen auf, wie nah die Störung (50 Hz) am Nutzsignal (40 Hz) liegt?\n"
        "3. Testen Sie den **Tiefpass-Filter**: Warum ist er für Alpha-Wellen super, zerstört aber die Gamma-Wellen? "
        "Welcher Filter rettet das Gamma-Signal?"
    )

    # Layout: Steuerung links (Sidebar oder Spalte), Grafik rechts
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Signal-Konfiguration")
        
        # 1. Patientenzustand / Frequenzbänder
        zustand = st.selectbox(
            "Patientenzustand (EEG-Grundrhythmus):",
            [
                "Tiefschlaf (Delta: ~2 Hz)", 
                "Entspannt / Augen geschlossen (Alpha: ~10 Hz)", 
                "Konzentriert / Aktiv (Beta: ~22 Hz)",
                "Höchste Konzentration / Fokus (Gamma: ~40 Hz)"
            ]
        )
        
        st.markdown("---")
        st.subheader("❌ Störquellen & Artefakte")
        
        # 2. Störungen und Artefakte als Checkboxen
        netzbrummen = st.checkbox("50 Hz Netzbrummen einschalten", value=False)
        blinzeln_aktiv = st.checkbox("👁️ Augenblinzeln (Artefakt) hinzufügen", value=False)

        st.markdown("---")
        st.subheader("🔍 Digitale Signalverarbeitung")
        
        # 3. Filter-Auswahl
        filter_typ = st.radio(
            "Digitalen Filter auswählen:",
            ["Kein Filter (Rohsignal)", "Tiefpass-Filter (Cutoff: 35 Hz)", "Hochpass-Filter (Cutoff: 5 Hz)", "Notch-Filter (50 Hz Schmalband)"]
        )

    with col2:
        st.subheader("📊 Signal-Anzeige (Oszilloskop)")

        # --- SIGNALGENERIERUNG (MATHEMATIK & NUMPY) ---
        fs = 500  # Abtastfrequenz in Hz
        dauer = 2.0  # Zeitdauer in Sekunden
        t = np.linspace(0, dauer, int(fs * dauer), endpoint=False)
        
        # Standard-Grundrauschen (weißes Rauschen)
        grundsignal = np.random.normal(0, 0.1, size=len(t))
        
        # Parameter je nach Patientenzustand setzen
        if "Delta" in zustand:
            # Delta: hohe Amplitude
            grundsignal += 1.5 * np.sin(2 * np.pi * 2.0 * t)
        elif "Alpha" in zustand:
            # Alpha: mittlere Amplitude
            grundsignal += 0.8 * np.sin(2 * np.pi * 10.0 * t)
        elif "Beta" in zustand:
            # Beta: geringe Amplitude
            grundsignal += 0.3 * np.sin(2 * np.pi * 22.0 * t)
        elif "Gamma" in zustand:
            # Gamma: sehr hohe Frequenz (~40 Hz), physiologisch sehr kleine Amplitude
            grundsignal += 0.15 * np.sin(2 * np.pi * 40.0 * t)

        # 50 Hz Netzbrummen überlagern
        if netzbrummen:
            grundsignal += 0.6 * np.sin(2 * np.pi * 50.0 * t)
            
        # Biologisches Artefakt (Augenblinzeln): Wird direkt über den Zustand der Checkbox gesteuert
        if blinzeln_aktiv:
            impuls_start = int(0.8 * fs)
            impuls_dauer = int(0.4 * fs)
            t_impuls = np.linspace(0, np.pi, impuls_dauer)
            # Sehr starke, langsame Auslenkung
            grundsignal[impuls_start:impuls_start+impuls_dauer] += 4.0 * np.sin(t_impuls)

        # --- DIGITALE FILTERUNG (SCIPY) ---
        verarbeitetes_signal = grundsignal.copy()
        
        if filter_typ == "Tiefpass-Filter (Cutoff: 35 Hz)":
            sos = signal.butter(4, 35, btype='low', fs=fs, output='sos')
            verarbeitetes_signal = signal.sosfiltfilt(sos, grundsignal)
            
        elif filter_typ == "Hochpass-Filter (Cutoff: 5 Hz)":
            sos = signal.butter(4, 5, btype='high', fs=fs, output='sos')
            verarbeitetes_signal = signal.sosfiltfilt(sos, grundsignal)
            
        elif filter_typ == "Notch-Filter (50 Hz Schmalband)":
            b, a = signal.iirnotch(50.0, 30.0, fs=fs)
            verarbeitetes_signal = signal.filtfilt(b, a, grundsignal)

        # --- VISUALISIERUNG (PLOTLY) ---
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=t, 
            y=verarbeitetes_signal, 
            mode='lines', 
            name='EEG Signal',
            line=dict(color='#2E7D32', width=2)
        ))
        
        fig.update_layout(
            xaxis_title="Zeit (Sekunden)",
            yaxis_title="Amplitude (µV)",
            yaxis=dict(range=[-3, 3]), # ANPASSUNG: Jetzt fest skaliert von -3 bis +3 µV
            margin=dict(l=40, r=40, t=10, b=40),
            height=400,
            showlegend=False,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Technische Zusatz-Informationen
        with st.expander("🔬 Signalverarbeitungs-Details einblenden"):
            st.markdown(
                f"**Aktueller Modus:** {filter_typ}\n\n"
                f"- **Abtastrate ($f_s$):** {fs} Hz\n"
                f"- **Filter-Methode:** `scipy.signal.sosfiltfilt` (Nullphasenfilterung)"
            )

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    eeg_applet()
