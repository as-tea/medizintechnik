import streamlit as st
import numpy as np
import plotly.graph_objects as go

def pacemaker_applet():
    st.header("⚡ Herzschrittmacher-Logik: NBG-Code")
    
    # Didaktischer Fokus für Clinical Engineers
    st.info(
        "Dieses Applet analysiert die Verarbeitungslogik von Schrittmachersystemen. "
        "Konfigurieren Sie den NBG-Code über die horizontalen Parameter. Beobachten Sie darunter, wie die einzelnen "
        "Einstellungen sich auf die Hardware-Kanäle und den Herzrhythmus auswirken."
    )

    st.subheader("🛠️ NBG-Code Konfigurator")
    
    # 1) NBG-Code Konfigurator quer über die Seite (4 Parameter nebeneinander)
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    
    with p_col1:
        pos1 = st.radio(
            "**Position I: Stimulationsort**",
            ["A (Atrium)", "V (Ventrikel)", "D (Dual: A+V)", "O (Keiner)"],
            index=1,
            help="Bestimmt, an welchem Ort im Herzmuskel die Stimulationsimpulse abgegeben werden."
        )
        
    with p_col2:
        pos2 = st.radio(
            "**Position II: Signalwahrnehmung (Sensing)**",
            ["A (Atrium)", "V (Ventrikel)", "D (Dual: A+V)", "O (Keiner)"],
            index=1,
            help="Bestimmt, an welchem Ort im Herzmuskel die spontane, eigene Herzaktivität gemessen wird."
        )
        
    with p_col3:
        pos3 = st.radio(
            "**Position III: Betriebsmodus**",
            ["I (Inhibition)", "T (Triggerung)", "D (Dual: I+T)", "O (Asynchron / Keine)"],
            index=0,
            help="Definiert die logische Verknüpfung: Setzt ein erkanntes Eigensignal den Timer zur Stimulation zurück (I) oder löst es eine Stimulation aus (T)?"
        )
        
    with p_col4:
        pos4 = st.radio(
            "**Position IV: Frequenzvariabilität**",
            ["O (Feste Basisfrequenz)", "R (Adaption der Stimulationsrate)"],
            index=0,
            help="Bestimmt, ob die Impulsfrequenz konstant bleibt, oder angepasst wird (z.B. bei Anstrengung)"
        )

    st.markdown("---")
    
    # 2) Darunter die Zusammenfassung der gewählten Konfiguration (Teil "Aktiver Betriebsmodus" wurde gelöscht)
    st.markdown("### Zusammenfassung und Bedeutung der Konfiguration")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Stimulationsort:**")
        if "A" in pos1 or "D" in pos1: st.write("✅ Stimulation im Atrium")
        if "V" in pos1 or "D" in pos1: st.write("✅ Stimulation im Ventrikel")
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
        st.success("Die Sensoren des Herzschrittmachers erkennen eine Änderung der Herzfrequenz und passen die Stimulation entsprechend an.")

    st.markdown("---")

    # Zweispaltiges Layout für Simulationseinstellungen und das Diagramm
    col_sim, col_graph = st.columns([1, 2])

    with col_sim:
        st.subheader("𫫛 Patientensignal-Simulation")
        patient_hr = st.slider("Intrinsische (eigene) Herzfrequenz (bpm):", min_value=40, max_value=100, value=55, step=5)
        pacemaker_base_rate = 60 # Feste programmierbare Untergrenze des Schrittmachers

    # Code-String für den Analyse-Block im Hintergrund zusammenbauen
    code_string = f"{pos1[0]}{pos2[0]}{pos3[0]}{pos4[0]}"

    with col_graph:
        # --- TIMING DIAGRAMM GENERIERUNG (PLOTLY) ---
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
        
        # Simulation der Timer-Zustände im Schrittmacher
        current_time = 0.0
        while current_time < 4.0:
            next_sensing_time = min([b for b in beat_times if b > last_action_time], default=99.0)
            
            if pos2[0] == "O":
                next_sensing_time = 99.0
                
            if next_sensing_time < last_action_time + pm_interval:
                current_time = next_sensing_time
                if pos3[0] in ["I", "D"]:
                    last_action_time = current_time
                else:
                    if last_action_time + pm_interval < 4.0:
                        pacing_times.append(last_action_time + pm_interval)
                        last_action_time += pm_interval
                    current_time = last_action_time
            else:
                if pos1[0] != "O":
                    if last_action_time + pm_interval < 4.0:
                        pacing_times.append(last_action_time + pm_interval)
                last_action_time += pm_interval
                current_time = last_action_time

        # Zeichne intrinsische Peaks (Gewebeaktivität)
        if pos2[0] != "O":
            for b in beat_times:
                idx = np.abs(t - b).argmin()
                y_heart[idx:idx+5] = 1.0
                
        # Zeichne Pacing Spikes (Hardware-Output)
        for p_time in pacing_times:
            idx = np.abs(t - p_time).argmin()
            y_pacing[idx:idx+3] = 1.5

        # Plotly Figur aufbauen
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y_heart, mode='lines', name='Biologisches Eigensignal', line=dict(color='#1E88E5', width=2)))
        fig.add_trace(go.Scatter(x=t, y=y_pacing, mode='lines', name='Schrittmacher-Impuls (Spike)', line=dict(color='#D81B60', width=2)))
        
        fig.update_layout(
            xaxis_title="Zeit Verlauf (Sekunden)",
            yaxis_title="Signalamplitude / Logik-Pegel",
            yaxis=dict(range=[-0.2, 2.0], tickvals=[0, 1, 1.5], ticktext=['Baseline', 'Eigenpotential', 'Pacing Spike']),
            margin=dict(l=40, r=20, t=20, b=40), height=260, template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- DIDAKTISCHER ANALYSIS-BLOCK ---
    st.markdown("---")
    with st.expander("Zusatzinfos für besseres Verständnis"):
        st.markdown(
            f"""
            ### Funktionelle Verhaltensanalyse für den Code **{code_string}**
            
            1. **Die Eingangsstufe (Sensing-Kanal):**
               Wenn Sie die intrinsische Herzfrequenz über den Regler auf **über 60 bpm** stellen, sehen Sie bei aktiver Position II und III (z.B. VVI), wie die roten Schrittmacher-Spikes komplett verschwinden. Der interne Komparator detektiert das Signal und **inhibiert (blockiert)** die Ausgangsstufe.
               
            2. **Der asynchrone Fehlerfall (z.B. VOO oder AOO):**
               Wählen Sie als Reaktionsmodus **O**. Unabhängig davon, wie schnell das biologische Herz schlägt, feuert die Ausgangsstufe starr im Takt der programmierten Frequenz. *Technisches Risiko:* Im schlimmsten Fall kann es zu Herzrhythmusstörungen kommen.
               
            3. **Blanking Period (Schutz vor Eigendestruktion):**
               Während eines Pacing Spikes (Anzeige im Diagramm: {pacing_times[:1] if pacing_times else 'Keiner'}) muss der Sensing-Eingang für ca. 20–40 ms komplett abgeschalten werden. Ohne diese Schaltung würde die Energie des eigenen Stimulationsimpulses die interne Elektronik überlasten.
            """
        )

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    pacemaker_applet()
