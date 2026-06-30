import streamlit as st
import numpy as np
import plotly.graph_objects as go

def spo2_applet():
    st.header("🩸 Pulsoximetrie: Absorptions- & Berechnungssynthesizer")
    
    # Didaktischer Begleittext
    st.info(
        "**Didaktischer Fokus – Wie misst der Sensor die Sättigung?**\n"
        "Verschieben Sie den $SpO_2$-Regler. Beobachten Sie im linken Diagramm, wie sich die Gesamt-Absorptionskurve "
        "des Blutes (grün) zwischen den reinen Kurven von sauerstoffarmem $Hb$ (blau) und sauerstoffreichem $HbO_2$ (rot) hin- und herverschiebt.\n\n"
        "Achten Sie auf die Schnittpunkte mit den Wellenlängen **660 nm** und **940 nm**: Dieses Verhältnis bestimmt, "
        "wie viel Licht den Detektor erreicht (rechtes Diagramm) und wie das Gerät daraus den $R$-Wert berechnet."
    )

    # Steuerung über einen prominenten Slider oben
    spo2 = st.slider("Eingestellte Sauerstoffsättigung ($SpO_2$ in %):", min_value=70, max_value=100, value=95, step=1)

    # Layout für die beiden Graphen nebeneinander
    col1, col2 = st.columns(2)

    # --- MATHEMATISCHE ANPASSUNG IN 50-NM-INTERVALLEN ---
    # Definition der exakten Stützstellen alle 50 nm gemeldet aus der Grafik
    wellenlaengen_stuetz = np.array([600, 650, 700, 750, 800, 850, 900, 950, 1000])
    
    # Manuelle Festlegung der molaren Extinktionskoeffizienten an den Stützstellen (abgelesen aus Ihrer Grafik)
    # Hb (Blau): Startet sehr hoch, fällt stark, kleiner Peak bei 750-800, flacht ab, sinkt am Ende drastisch
    hb_stuetz = np.array([10.0, 4.0, 1.8, 1.2, 0.7, 0.6, 0.7, 0.5, 0.12])
    
    # HbO2 (Rot): Startet moderat, hat ein Minimum bei ~700, steigt im Infrarotbereich an, bildet ein Plateau
    hbo2_stuetz = np.array([2.5, 0.4, 0.25, 0.45, 0.8, 1.0, 1.2, 1.2, 0.9])

    # Für einen schönen, glatten Plot interpolieren wir zwischen den 50-nm-Schritten
    wellenlaengen = np.linspace(600, 1000, 400)
    abs_hb = np.interp(wellenlaengen, wellenlaengen_stuetz, hb_stuetz)
    abs_hbo2 = np.interp(wellenlaengen, wellenlaengen_stuetz, hbo2_stuetz)

    # Die aktuelle Gesamtabsorption ist die gewichtete Mischung basierend auf SpO2
    fraktion_hbo2 = spo2 / 100.0
    fraktion_hb = 1.0 - fraktion_hbo2
    abs_gesamt = (fraktion_hb * abs_hb) + (fraktion_hbo2 * abs_hbo2)

    # Spezifische Werte an den Messpunkten 660nm und 940nm berechnen (wird aus den interpolierten Kurven exakt extrahiert)
    idx_660 = np.abs(wellenlaengen - 660).argmin()
    idx_940 = np.abs(wellenlaengen - 940).argmin()
    
    abs_at_660 = abs_gesamt[idx_660]
    abs_at_940 = abs_gesamt[idx_940]

    with col1:
        st.subheader("📈 Optisches Spektrum & Absorption")
        
        fig_spec = go.Figure()
        
        # Basis-Kurven (Hb und HbO2)
        fig_spec.add_trace(go.Scatter(x=wellenlaengen, y=abs_hb, mode='lines', name='Deoxygeniertes Hb', line=dict(color='#1976D2', width=2)))
        fig_spec.add_trace(go.Scatter(x=wellenlaengen, y=abs_hbo2, mode='lines', name='Oxygeniertes HbO₂', line=dict(color='#D32F2F', width=2)))
        
        # Dynamische Gesamtaborptionskurve
        fig_spec.add_trace(go.Scatter(x=wellenlaengen, y=abs_gesamt, mode='lines', name='Gesamt-Blutabsorption', line=dict(color='#2E7D32', width=3)))
        
        # Vertikale Parallelen bei den Wellenlängen des echten Sensors
        fig_spec.add_vline(x=660, line_width=2, line_dash="dash", line_color="red")
        fig_spec.add_vline(x=940, line_width=2, line_dash="dash", line_color="purple")
        
        # Schnittpunkte als markante Punkte
        fig_spec.add_trace(go.Scatter(x=[660], y=[abs_at_660], mode='markers', marker=dict(color='red', size=10, symbol='circle'), name='Messpunkt 660 nm'))
        fig_spec.add_trace(go.Scatter(x=[940], y=[abs_at_940], mode='markers', marker=dict(color='purple', size=10, symbol='circle'), name='Messpunkt 940 nm'))
        
        fig_spec.update_layout(
            xaxis_title="Wellenlänge (nm)",
            yaxis_title="Molarer Extinktionskoeffizient",
            xaxis=dict(range=[600, 1000]),
            # ANPASSUNG DER LOG-ACHSE: Lesbare Dezimalzahlen statt Potenzen
            yaxis=dict(
                type='log', 
                range=[-1.0, 1.1],
                tickvals=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
                ticktext=['0.1', '0.2', '0.5', '1.0', '2.0', '5.0', '10.0']
            ),
            margin=dict(l=40, r=40, t=10, b=40),
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            template="plotly_white"
        )
        
        st.plotly_chart(fig_spec, use_container_width=True)

    with col2:
        st.subheader("📉 Signal am Detektor (Transmission)")
        
        t = np.linspace(0, 2, 200)
        puls = 0.1 * np.sin(2 * np.pi * 1.2 * t)
        
        # Intensität antiproportional zur Absorption (Lambert-Beer Näherung)
        intensitaet_base_660 = 2.5 * np.exp(-abs_at_660 * 0.4)
        intensitaet_base_940 = 2.5 * np.exp(-abs_at_940 * 0.4)
        
        sig_660 = intensitaet_base_660 - (0.05 * intensitaet_base_660 * puls)
        sig_940 = intensitaet_base_940 - (0.05 * intensitaet_base_940 * puls)
        
        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(x=t, y=sig_660, mode='lines', name='Detektorsignal Rot (660 nm)', line=dict(color='red', width=2)))
        fig_sig.add_trace(go.Scatter(x=t, y=sig_940, mode='lines', name='Detektorsignal IR (940 nm)', line=dict(color='purple', width=2)))
        
        fig_sig.update_layout(
            xaxis_title="Zeit (Sekunden)",
            yaxis_title="Empfangene Lichtintensität (V)",
            yaxis=dict(range=[0, 3.5]),
            margin=dict(l=40, r=40, t=10, b=40),
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            template="plotly_white"
        )
        
        st.plotly_chart(fig_sig, use_container_width=True)

    # --- MATHEMATISCHE AUSWERTUNG ---
    st.markdown("---")
    st.subheader("📊 Berechnung der Sauerstoffsättigung aus den Kurvenwerten")
    
    r_wert_berechnet = abs_at_660 / abs_at_940
    
    c1, c2, c3 = st.columns(3)
    c1.metric(label="Absorption bei 660 nm (Rot)", value=f"{abs_at_660:.3f}")
    c2.metric(label="Absorption bei 940 nm (Infrarot)", value=f"{abs_at_940:.3f}")
    c3.metric(label="Berechneter Verhältniswert (R-Wert)", value=f"{r_wert_berechnet:.3f}")
    
    with st.expander("🔬 Der mathematische Rechenweg für Studierende"):
        st.markdown(
            r"""
            Ein Pulsoximeter berechnet die Sättigung nicht absolut, sondern misst das Verhältnis der Dämpfungen bei beiden Wellenlängen.
            
            Daraus ergibt sich der **$R$-Wert (Ratio-of-Ratios)**:
            """
            f"$$R = \\frac{{\\text{{Absorption}}_{{660nm}}}}{{\\text{{Absorption}}_{{940nm}}}} = \\frac{{{abs_at_660:.3f}}}{{{abs_at_940:.3f}}} = {r_wert_berechnet:.3f}$$"
            """
            **Der Merksatz für die Prüfung (anhand der Kurven nachvollziehbar):**
            - Bei **hoher Sättigung (100% $SpO_2$)** dominiert $HbO_2$ (rote Kurve). Betrachten Sie die Schnittpunkte: Bei $940\,\text{nm}$ ist die Absorption höher als bei $660\,\text{nm}$. Der Zähler ist kleiner als der Nenner $\rightarrow$ Der **$R$-Wert ist klein (< 1.0)**.
            - Bei **niedriger Sättigung ($70\%\,SpO_2$)** gewinnt das ungesättigte $Hb$ (blaue Kurve) die Ober
