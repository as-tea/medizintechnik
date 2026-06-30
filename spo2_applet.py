import streamlit as st
import numpy as np
import plotly.graph_objects as go

def spo2_applet():
    st.header("🩸 Der interaktive Pulsoximeter-Sensor")
    
    # Didaktischer Begleittext am Anfang
    st.info(
        "**Arbeitsaufgabe für Studierende:**\n"
        "1. Stellen Sie die Sauerstoffsättigung ($SpO_2$) auf **100%** und beobachten Sie die Amplituden der roten (660 nm) und infraroten (940 nm) Lichtkurve.\n"
        "2. Reduzieren Sie den $SpO_2$-Wert schrittweise auf **70%**. Wie verändern sich die pulsierenden Anteile zueinander und was passiert mit dem berechneten Verhältniswert ($R$)?\n"
        "3. Aktivieren Sie die Fehlerquelle **'Kalte Finger'**. Warum kann das Gerät nun die Sättigung nur noch schwer oder gar nicht mehr berechnen?\n"
        "4. Aktivieren Sie **'Fremdlicht'**. Welcher Lichtanteil (der konstante Basis-Anteil oder der pulsierende Anteil) verschiebt sich dadurch?"
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
        R_wert = (110 - spo2) / 25.0
        
        # Zeitachse definieren
        fs = 200  # Abtastfrequenz in Hz
        dauer = 3.0  # 3 Sekunden Anzeige
        t = np.linspace(0, dauer, int(fs * dauer), endpoint=False)
        
        # Pulsfrequenz in Hz umrechnen
        f_puls = hf / 60.0
        
        # Basis-Pulsform simulieren (Grundwelle + harmonische Oberschwingung für die dikrote Welle)
        puls_form = (np.sin(2 * np.pi * f_puls * t) * 0.7 + 
                     np.sin(2 * np.pi * 2 * f_puls * t - 1.0) * 0.3)
        
        # Rauschen hinzufügen
        rauschen = np.random.normal(0, 0.01, size=len(t))

        # --- INITIALISIERUNG DER LICHTANTEILE ---
        # Infrarot (940 nm) als stabile Referenz setzen
        konstant_ir = 1.0
        pulsierend_ir = 0.06  # 6% des Lichts pulsiert idealerweise
        
        # Rot (660 nm) wird über den R-Wert bestimmt
        konstant_rot = 1.0
        pulsierend_rot = R_wert * (pulsierend_ir / konstant_ir) * konstant_rot

        # --- EFFEKTE DER FEHLERQUELLEN ---
        fehlermeldung = ""
        if fehler == "Kalte Finger (Vasokonstriktion / schwacher Puls)":
            # Reduziert den pulsierenden Anteil massiv (weniger durchblutetes Gewebe)
            pulsierend_ir *= 0.15
            pulsierend_rot *= 0.15
            fehlermeldung = "⚠️ Warnung: Pulsierender Signalanteil zu gering (Puls wird kaum erkannt)!"
        elif fehler == "Fremdlicht-Einstrahlung (z.B. OP-Leuchte)":
            # Fügt dem konstanten Basis-Lichtanteil ein starkes Störsignal hinzu
            konstant_ir += 0.8
            konstant_rot += 1.2
            fehlermeldung = "⚠️ Warnung: Konstanter Lichtanteil durch Fremdlicht verschoben (Messung verfälscht)!"

        # --- SIGNAL-SYNTHESE (Transmittiertes Licht an der Fotodiode) ---
        # Mehr Absorption durch Blut = weniger Licht kommt an der Fotodiode an
        signal_ir = konstant_ir - (pulsierend_ir * puls_form) + rauschen
        signal_rot = konstant_rot - (pulsierend_rot * puls_form) + rauschen

        # --- PLOTLY GRAPH ---
        fig = go.Figure()
        
        # Rote Kurve (660 nm)
        fig.add_trace(go.Scatter(
            x=t, y=signal_rot, mode='lines', 
            name='Rotes Licht (660 nm)', line=dict(color='#D32F2F', width=2.5)
        ))
        
        # Infrarote Kurve (940 nm)
        fig.add_trace(go.Scatter(
            x=t, y=signal_ir, mode='lines', 
            name='Infrarotes Licht (940 nm)', line=dict(color='#1976D2', width=2.5, dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title="Zeit (Sekunden)",
            yaxis_title="Lichtintensität am Detektor (V)",
            yaxis=dict(range=[0.4, 2.5]), 
            margin=dict(l=40, r=40, t=10, b=40),
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        if fehlermeldung:
            st.warning(fehlermeldung)

        # --- MATHEMATISCHE AUSWERTUNG & KALIBRATION ---
        st.markdown("---")
        st.subheader("📊 Optische Auswertung (Ratio-of-Ratios)")
        
        # Live-Berechnung des R-Wertes aus den gewählten Anteilen
        aktueller_r_effektiv = (pulsierend_rot / konstant_rot) / (pulsierend_ir / konstant_ir)
        
        # Anzeige in Kacheln
        calc_col1, calc_col2, calc_col3 = st.columns(3)
        calc_col1.metric(label="Eingestellte SpO₂", value=f"{spo2} %")
        calc_col2.metric(label="Berechneter Verhältniswert (R)", value=f"{aktueller_r_effektiv:.3f}")
        
        if fehler == "Fremdlicht-Einstrahlung (z.B. OP-Leuchte)":
            falsche_spo2 = 110 - 25 * aktueller_r_effektiv
            calc_col3.metric(label="Vom Gerät gemessene SpO₂", value=f"{falsche_spo2:.1f} %", delta=f"{falsche_spo2 - spo2:.1f} % Fehler", delta_color="inverse")
        else:
            calc_col3.metric(label="Vom Gerät gemessene SpO₂", value=f"{spo2} % (Korrekt)")

        # Didaktische Zusatz-Erklärung ohne E-Technik-Begriffe
        with st.expander("🔬 Theorie-Hintergrund: Optische Gewebeabsorption"):
            st.markdown(
                r"""
                Die Pulsoximetrie macht sich das **Lambert-Beersche Gesetz** zunutze. Wenn Licht durch den Finger gesendet wird, unterscheidet das Gerät zwei Anteile:
                
                1. **Der konstante Lichtanteil:** Gewebe, Knochen, Haut und das ruhige venöse Blut absorbieren immer gleich viel Licht. Dieser Anteil ändert sich während des Herzschlags nicht.
                2. **Der pulsierende Lichtanteil:** Mit jedem Herzschlag (Systole) wird neues arterielles Blut in die Fingerspitze gepumpt. Das Gefäß dehnt sich kurz aus, wodurch mehr Licht geschluckt wird. Das sorgt für das rhythmische Auf und Ab im Diagramm.
                
                Da sauerstoffreiches Blut ($HbO_2$) und sauerstoffarmes Blut ($Hb$) rotes und infrarotes Licht völlig unterschiedlich absorbieren, 
                setzt das Gerät die Signalanteile beider Wellenlängen zueinander in Beziehung ($R$-Wert):
                """
                f"$$R = \\frac{{\\text{{(Pulsierender Anteil / Konstanter Anteil)}}_{{rot}}}}{{\\text{{(Pulsierender Anteil / Konstanter Anteil)}}_{{infrarot}}}} = \\frac{{{pulsierend_rot:.3f} / {konstant_rot:.1f}}}{{{pulsierend_ir:.3f} / {konstant_ir:.1f}}} = {aktueller_r_effektiv:.3f}$$"
                """
                - **Hohe Sättigung (100%):** Infrarot wird stark absorbiert, Rot kaum $\rightarrow$ Der $R$-Wert ist klein (~0.4).
                - **Niedrige Sättigung (70%):** Rot wird massiv absorbiert $\rightarrow$ Der $R$-Wert wird groß (~1.6).
                """
            )

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    spo2_applet()
