import streamlit as st
import numpy as np
import plotly.graph_objects as go

def spo2_applet():
    st.header("🩸 Der interaktive Pulsoximeter-Sensor")
    
    # Didaktischer Begleittext am Anfang
    st.info(
        "**Arbeitsaufgabe für Studierende:**\n"
        "1. Stellen Sie die Sauerstoffsättigung ($SpO_2$) auf **100%** und beobachten Sie die Amplituden der roten (660 nm) und infraroten (940 nm) PPG-Kurve.\n"
        "2. Reduzieren Sie den $SpO_2$-Wert schrittweise auf **70%**. Wie verändern sich die Amplituden zueinander und was passiert mit dem berechneten $R$-Wert?\n"
        "3. Aktivieren Sie die Fehlerquelle **'Kalte Finger'**. Warum kann das Gerät nun die Sättigung nur noch schwer oder gar nicht mehr berechnen?\n"
        "4. Aktivieren Sie **'Fremdlicht'**. Welcher Signalanteil (AC oder DC) verschiebt sich dadurch?"
    )

    # Layout: Steuerung links, Grafik und Auswertung rechts
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Patienten- & Sensor-Setup")
        
        # 1. Regler für Vitalwerte
        spo2 = st.slider("Sauerstoffsättigung (SpO₂ in %):", min_value=70, max_value=100, value=98, step=1)
        hf = st.slider("Pulsfrequenz (HF in bpm):", min_value=40, max_value=140, value=75, step=5)
        
        st.markdown("---")
        st.subheader("⚠️ Klinische Störquellen")
        
        # 2. Fehlerquellen-Auswahl
        fehler = st.selectbox(
            "Messtechnische Fehlerquelle simulieren:",
            ["Kein Fehler (Ideale Messung)", "Kalte Finger (Vasokonstriktion / schwacher Puls)", "Fremdlicht-Einstrahlung (z.B. OP-Leuchte)"]
        )

    with col2:
        st.subheader("📊 Fotodiode: Signal-Anzeige (PPG)")

        # --- REALE PHYSIKALISCHE MODELLIERUNG ---
        # Kalibrationskurve eines typischen Pulsoximeters: SpO2 = 110 - 25 * R
        # Daraus berechnen wir den theoretischen R-Wert (Ratio-of-Ratios)
        R_wert = (110 - spo2) / 25.0
        
        # Zeitachse definieren
        fs = 200  # Abtastfrequenz in Hz
        dauer = 3.0  # 3 Sekunden Anzeige
        t = np.linspace(0, dauer, int(fs * dauer), endpoint=False)
        
        # Pulsfrequenz in Hz umrechnen
        f_puls = hf / 60.0
        
        # Basis-Pulsform simulieren (Grundwelle + harmonische Oberschwingung für die dikrote Welle/Inzisur)
        puls_form = (np.sin(2 * np.pi * f_puls * t) * 0.7 + 
                     np.sin(2 * np.pi * 2 * f_puls * t - 1.0) * 0.3)
        
        # Rauschen hinzufügen
        rauschen = np.random.normal(0, 0.01, size=len(t))

        # --- INITIALISIERUNG AC UND DC ANTEILE ---
        # Infrarot (940 nm) als stabile Referenz setzen
        dc_ir = 1.0
        ac_ir = 0.06  # 6% des DC-Signals pulsiert idealerweise
        
        # Rot (660 nm) wird über den R-Wert bestimmt: R = (AC_rot/DC_rot) / (AC_ir/DC_ir)
        dc_rot = 1.0
        ac_rot = R_wert * (ac_ir / dc_ir) * dc_rot

        # --- EFFEKTE DER FEHLERQUELLEN ---
        fehlermeldung = ""
        if fehler == "Kalte Finger (Vasokonstriktion / schwacher Puls)":
            # Reduziert den pulsierenden AC-Anteil massiv (weniger durchblutetes Gewebe)
            ac_ir *= 0.15
            ac_rot *= 0.15
            fehlermeldung = "⚠️ Warnung: AC-Amplitude zu gering (Puls wird kaum erkannt)!"
        elif fehler == "Fremdlicht-Einstrahlung (z.B. OP-Leuchte)":
            # Fügt dem DC-Anteil ein starkes konstantes Lichtsignal (Offset) hinzu
            dc_ir += 0.8
            dc_rot += 1.2
            fehlermeldung = "⚠️ Warnung: DC-Offset verschoben (R-Wert Berechnung verfälscht)!"

        # --- SIGNAL-SYNTHESE (Transmittiertes Licht an der Fotodiode) ---
        # Wichtig: Mehr Absorption durch Blut = weniger Licht kommt an der Fotodiode an (Kurve geht nach unten)
        signal_ir = dc_ir - (ac_ir * puls_form) + rauschen
        signal_rot = dc_rot - (ac_rot * puls_form) + rauschen

        # --- PLOTLY GRAPH ---
        fig = go.Figure()
        
        # Rote Kurve (660 nm)
        fig.add_trace(go.Scatter(
            x=t, y=signal_rot, mode='lines', 
            name='Rotes Licht (660 nm)', line=dict(color='#D32F2F', width=2.
