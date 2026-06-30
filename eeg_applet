import streamlit as st
import numpy as np
import scipy.signal as signal
import plotly.graph_objects as go

def eeg_applet():
    st.header("🧠 Der Gehirnwellen-Synthesizer & Filter")
    
    # Didaktischer Begleittext am Anfang
    st.info(
        "**Arbeitsaufgabe für Studierende:**\n"
        "1. Wählen Sie den Zustand **'Entspannt / Augen geschlossen'** (Alpha-Wellen).\n"
        "2. Schalten Sie das **50 Hz Netzbrummen** ein und fügen Sie ein **Augenblinzeln** hinzu. Betrachten Sie das gestörte Signal.\n"
        "3. Testen Sie die verschiedenen Filtertypen: Welcher Filter entfernt das Netzbrummen, ohne die zugrundeliegenden Alpha-Wellen zu dämpfen? "
        "Und welcher Filter hilft gegen das Blinzartefakt?"
    )

    # Layout: Steuerung links (Sidebar oder Spalte), Grafik rechts
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Signal-Konfiguration")
        
        # 1. Patientenzustand / Frequenzbänder
        zustand = st.selectbox(
            "Patientenzustand (EEG-Grundrhythmus):",
            ["Tiefschlaf (Delta)", "Entspannt / Augen geschlossen (Alpha)", "Konzentriert / Aktiv (Beta)"]
        )
        
        st.markdown("---")
        st.subheader("❌ Störquellen & Artefakte")
        
        # 2. Störungen
        netzbrummen = st.checkbox("50 Hz Netzbrummen einschalten", value=False)
        
        # Da Streamlit bei jedem Klick neu lädt, nutzen wir den Session State, 
        # um das Blinzartefakt bei Button-Druck für einen Durchlauf zu aktivieren.
        if "blinzeln_aktiv" not in st.state:
            st.state.blinzeln_aktiv = False
            
        if st.button("👁️ Einmaliges Augenblinzeln simulieren"):
            st.state.blinzeln_aktiv = True

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
        fs = 500  # Abtastfrequenz in Hz (500 Samples pro Sekunde)
        dauer = 2.0  # Zeitdauer in Sekunden
        t = np.linspace(0, dauer, int(fs * dauer), endpoint=False)
        
        # Standard-Grundrauschen (weißes Rauschen)
        grundsignal = np.random.normal(0, 0.1, size=len(t))
        
        # Parameter je nach Patientenzustand setzen
        if "Delta" in zustand:
            # Delta-Wellen: 0.5 - 4 Hz (wir nehmen ca. 2 Hz), hohe Amplitude
            grundsignal += 1.5 * np.sin(2 * np.pi * 2.0 * t)
        elif "Alpha" in zustand:
            # Alpha-Wellen: 8 - 13 Hz (wir nehmen ca. 10 Hz), mittlere Amplitude
            grundsignal += 0.8 * np.sin(2 * np.pi * 10.0 * t)
        elif "Beta" in zustand:
            # Beta-Wellen: 14 - 30 Hz (wir nehmen ca. 20 Hz), geringe Amplitude
            grundsignal += 0.3 * np.sin(2 * np.pi * 22.0 * t)

        # 50 Hz Netzbrummen überlagern
        if netzbrummen:
            grundsignal += 0.6 * np.sin(2 * np.pi * 50.0 * t)
            
        # Biologisches Artefakt (Augenblinzeln): Ein starker, langsamer Halbsinus-Impuls
        if st.state.blinzeln_aktiv:
            # Impuls in der Mitte des Signals platzieren (von Sekunde 0.8 bis 1.2)
            impuls_start = int(0.8 * fs)
            impuls_dauer = int(0.4 * fs)
            t_impuls = np.linspace(0, np.pi, impuls_dauer)
            # Sehr starke Auslenkung (typisch für EOG-Artefakte im Stirn-EEG)
            grundsignal[impuls_start:impuls_start+impuls_dauer] += 4.0 * np.sin(t_impuls)
            # Zustand für den nächsten Durchlauf zurücksetzen
            st.state.blinzeln_aktiv = False

        # --- DIGITALE FILTERUNG (SCIPY) ---
        verarbeitetes_signal = grundsignal.copy()
        
        # Filter-Implementierungen mit Butterworth-Filterkoeffizienten (sos-Format für Stabilität)
        if filter_typ == "Tiefpass-Filter (Cutoff: 35 Hz)":
            # Lässt Frequenzen < 35 Hz durch, dämpft 50 Hz Brummen
            sos = signal.butter(4, 35, btype='low', fs=fs, output='sos')
            verarbeitetes_signal = signal.sosfiltfilt(sos, grundsignal)
            
        elif filter_typ == "Hochpass-Filter (Cutoff: 5 Hz)":
            # Lässt Frequenzen > 5 Hz durch, dämpft langsame Blinzartefakte (und Delta-Wellen!)
            sos = signal.butter(4, 5, btype='high', fs=fs, output='sos')
            verarbeitetes_signal = signal.sosfiltfilt(sos, grundsignal)
            
        elif filter_typ == "Notch-Filter (50 Hz Schmalband)":
            # Entfernt gezielt nur einen schmalen Bereich um 50 Hz
            b, a = signal.iirnotch(50.0, 30.0, fs=fs)
            verarbeitetes_signal = signal.filtfilt(b, a, grundsignal)

        # --- VISUALISIERUNG (PLOTLY) ---
        fig = go.Figure()
        
        # Plot des resultierenden Signals
        fig.add_trace(go.Scatter(
            x=t, 
            y=verarbeitetes_signal, 
            mode='lines', 
            name='EEG Signal',
            line=dict(color='#2E7D32', width=2)
        ))
        
        # Diagramm-Styling für Medizin-Dashboard-Look
        fig.update_layout(
            xaxis_title="Zeit (Sekunden)",
            yaxis_title="Amplitude (µV)",
            yaxis=dict(range=[-6, 6]), # Feste Skalierung, damit Amplitudenänderungen auffallen
            margin=dict(l=40, r=40, t=10, b=40),
            height=400,
            showlegend=False,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Technische Zusatz-Informationen für Studierende
        with st.expander("🔬 Signalverarbeitungs-Details einblenden"):
            st.markdown(
                f"**Aktueller Modus:** {filter_typ}\n\n"
                f"- **Abtastrate ($f_s$):** {fs} Hz (Abtasttheorem eingehalten für alle Gehirnwellen & Netzbrummen)\n"
                f"- **Filter-Methode:** `scipy.signal.sosfiltfilt` (Nullphasenfilterung verhindert Phasenverschiebung im Signal)"
            )

# Ermöglicht das direkte Testen/Ausführen des Skripts
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    eeg_applet()
