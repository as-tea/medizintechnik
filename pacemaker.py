import streamlit as st
import numpy as np
import plotly.graph_objects as go

def pacemaker_applet():
    st.header("⚡ Herzschrittmacher-Logik: NBG-Code")
    
    # Didaktischer Fokus für Clinical Engineers
    st.info(
        "Dieses Applet analysiert die Verarbeitungslogik von Schrittmachersystemen. "
        "Konfigurieren Sie den NBG-Code auf der linken Seite. Beobachten Sie rechts, wie die einzelnen Parameter "
        "sich auf den Herzrhythmus auswirken."
    )

    # Layout: Steuerung links, technische Analyse rechts
    col_control, col_display = st.columns([1, 2])

    with col_control:
        st.subheader("🛠️ NBG-Code Konfigurator")
        
        # Position 1: Pacing
        pos1 = st.radio(
            "**Position I: Stimulationsort**",
            ["A (Atrium)", "V (Ventrikel)", "D (Dual: A+V)", "O (Keiner)"],
            index=1,
            help="Bestimmt, an welchem Ort im Herzmuskel die Stimulationsimpulse abgegeben werden."
        )
        
        # Position 2: Sensing
        pos2 = st.radio(
            "**Position II: Signalwahrnehmung (Sensing)**",
            ["A (Atrium)", "V (Ventrikel)", "D (Dual: A+V)", "O (Keiner)"],
            index=1,
            help="Bestimmt, an welchem Ort im Herzmuskel die spontane, eigene Herzaktivität gemessen wird."
        )
        
        # Position 3: Reaktion
        pos3 = st.radio(
            "**Position III: Art der Steuerung (Betriebsmodus)**",
            ["I (Inhibition)", "T (Triggerung)", "D (Dual: I+T)", "O (Asynchron / Keine)"],
            index=0,
            help="Definiert die logische Verknüpfung: Setzt ein erkanntes Eigensignal den Timer zur Stimulation zurück (I) oder löst es eine Stimulation aus (T)?"
        )
        
        # Position 4: Frequenzanpassung
        pos4 = st.radio(
            "**Position IV: Frequenzvariabilität**",
            ["O (Feste Basisfrequenz)", "R (Adaption der Stimulationsrate)"],
            index=0,
            help="Bestimmt, ob die Impulsfrequenz konstant bleibt, oder angepasst wird (z.B. bei Anstrengung)"
        )

        st.markdown("---")
        st.subheader("🫀 Patientensignal-Simulation")
        patient_hr = st.slider("Intrinsische (eigene) Herzfrequenz (bpm):", min_value=40, max_value=100, value=55, step=5)
        pacemaker_base_rate = 60 # Feste programmierbare Untergrenze des Schrittmachers

    # Code-String zusammenbauen (nur die Anfangsbuchstaben)
    code_string = f"{pos1[0]}{pos2[0]}{pos3[0]}{pos4[0]}"

    with col_display:
        st.subheader(f"📟 Aktiver Betriebsmodus: {code_string}")
        
        # Technische Übersetzung der Hardware-Kanäle
        st.markdown("### 🔍 Hardware- & Firmware-Konfiguration")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Stimulationsort:**")
            if "A" in pos1 or "D" in pos1: st.write("✅ Stimulation im Atrum")
            if "V" in pos1 or "D" in pos1: st.write("✅ Stimuliation im Ventrikel")
            if "O" in pos1: st.write("❌ Ausgänge deaktiviert")
            
        with c2:
            st.markdown("**Ort des Sensings:**")
            if "A" in pos2 or "D" in pos2: st.write("🔍 Atriale P-Wellen-Detektion")
            if "V" in pos2 or "D" in pos2: st.write("🔍 Ventrikuläre R-Zackendetektion")
            if "O" in pos2: st.write("❌ Sensorik stummgeschaltet")
            
        with c3:
            st.markdown("**Interne Logik:**")
            if "I" in pos3 or "D" in pos3: st.write("⏱️ Herzaktivität startet den Timer bis zum Auslösen einer Stimulation neu")
            if "T" in pos3 or "D" in pos3: st.write("⚡ Herzaktivität führt zum Auslösen einer Stimulation")
            if "O" in pos3: st.write("🔄 Starres Stimulationsmuster, unabhängig von eigener Herzaktivität")

        if "R" in pos4:
            st.success("🤖 Die Sensoren des Herzschrittmachers erkennen eine Änderung der Herzfrequenz und passen die Stimulation entsprechend an.")

        # --- TIMING DIAGRAMM GENERIERUNG (PLOTLY) ---
        st.markdown("---")
        st.markdown("### 📉 Technisches Timing- & Aktivitätsdiagramm")
        
        t = np.linspace(0, 4, 400)
        y_heart = np.zeros_like(t)
        y_pacing = np.zeros_like(t)
        
        # Berechne Intervalle in Sekunden
        intrinsic_interval = 60.0 / patient_hr
        pm_interval = 60.0 / pacemaker_base_rate
        
        # Erzeuge intrinsische Herzaktionen (R-Zacken)
        num_beats = int(4.0 / intrinsic_interval) + 1
        beat_times = [i * intrinsic_interval + 0.2 for i in range(num_beats) if i * intrinsic_interval + 0.2 < 4.0]
        
        # Schrittmacher-Logik anwenden
        pacing_times = []
        last_action_time = 0.0
        
        # Vereinfachte Schleife zur Simulation der Timer-Zustände im Schrittmacher
        current_time = 0.0
        while current_time < 4.0:
            # Prüfen, ob ein Sensing-Ereignis vor dem Timer-Ablauf stattfindet
            next_sensing_time = min([b for b in beat_times if b > last_action_time], default=99.0)
            
            # Hat das Gerät überhaupt Sensing aktiviert?
            if pos2[0] == "O":
                next_sensing_time = 99.0 # Ignoriere Eigensignale komplett
                
            if next_sensing_time < last_action_time + pm_interval:
                # SENSING-EREIGNIS TRITT EIN
                current_time = next_sensing_time
                if pos3[0] in ["I", "D"]:
                    # Inhibition: Timer wird zurückgesetzt, kein Spike!
                    last_action_time = current_time
                else:
                    # Asynchroner Modus (O): Eigensignal wird ignoriere, Timer läuft starr weiter
                    if last_action_time + pm_interval < 4.0:
                        pacing_times.append(last_action_time + pm_interval)
                        last_action_time += pm_interval
                    current_time = last_action_time
            else:
                # TIMER LÄUFT AB -> Schrittmacher muss stimulieren (sofern Ausgänge aktiv)
                if pos1[0] != "O":
                    if last_action_time + pm_interval < 4.0:
                        pacing_times.append(last_action_time + pm_interval)
                last_action_time += pm_interval
                current_time = last_action_time

        # Zeichne intrinsische Peaks (Gewebeaktivität)
        if pos2[0] != "O":
            for b in beat_times:
                idx = np.abs(t - b).argmin()
                y_heart[idx:idx+5] = 1.0 # Eigensignal-Peak
                
        # Zeichne Pacing Spikes (Hardware-Output)
        for p_time in pacing_times:
            idx = np.abs(t - p_time).argmin()
            y_pacing[idx:idx+3] = 1.5 # Deutlicher, scharfer Schrittmacher-Spike

        # Plotly Figur aufbauen
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y_heart, mode='lines', name='Biologisches Eigensignal', line=dict(color='#1E88E5', width=2)))
        fig.add_trace(go.Scatter(x=t, y=y_pacing, mode='lines', name='Schrittmacher-Impuls (Spike)', line=dict(color='#D81B60', width=2)))
        
        # Schwellenwert-Linien für Blanking/Refraktärzeit-Andeutung
        fig.update_layout(
            xaxis_title="Zeit Verlauf (Sekunden)",
            yaxis_title="Signalamplitude / Logik-Pegel",
            yaxis=dict(range=[-0.2, 2.0], tickvals=[0, 1, 1.5], ticktext=['Baseline', 'Eigenpotential', 'Pacing Spike']),
            margin=dict(l=40, r=20, t=20, b=40), height=300, template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- DIDAKTISCHER ANALYSIS-BLOCK ---
    st.markdown("---")
    with st.expander("🔬 Firmware- & Schaltungsanalyse für Clinical Engineers"):
        st.markdown(
            f"""
            ### Funktionelle Verhaltensanalyse für den Code **{code_string}**
            
            1. **Die Eingangsstufe (Sensing-Kanal):**
               Wenn Sie die intrinsische Herzfrequenz über den Regler auf **über 60 bpm** stellen, sehen Sie bei aktiver Position II und III (z.B. VVI), wie die roten Schrittmacher-Spikes komplett verschwinden. Der interne Komparator detektiert das Signal, überschreitet die konfigurierte mV-Schwelle und **inhibiert (blockiert)** die Ausgangsstufe.
               
            2. **Der asynchrone Fehlerfall (z.B. VOO oder AOO):**
               Wählen Sie als Reaktionsmodus **O**. Unabhängig davon, wie schnell das biologische Herz schlägt, feuert die Ausgangsstufe starr im Takt der programmierten Frequenz. *Technisches Risiko:* Fällt ein solcher ungesteuerter Spike exakt in die vulnerable Phase der T-Welle des Eigensignals, kann dies fatale Rhythmusstörungen auslösen.
               
            3. **Blanking Period (Schutz vor Eigendestruktion):**
               Während eines Pacing Spikes (Anzeige im Diagramm: {pacing_times[:1] if pacing_times else 'Keiner'}) muss der Sensing-Eingang für ca. 20–40 ms komplett **kurzgeschlossen (geblankt)** werden. Ohne diese Schaltung würde die gewaltige Energie des eigenen Stimulationsimpulses den empfindlichen Sensing-Verstärker überlasten oder fälschlicherweise als biologisches Eigensignal interpretieren (Oversensing).
            """
        )

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    pacemaker_applet()
