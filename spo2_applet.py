import streamlit as st
import numpy as np
import plotly.graph_objects as go

def spo2_applet():
    st.header("🩸 Pulsoximetrie: Absorptions- & Berechnungs")
    
    # Didaktischer Begleittext
    st.info(
        "**Wie misst der Sensor die Sättigung?**\n\n"
        "Verschieben Sie den $SpO_2$-Regler. Beobachten Sie im linken Diagramm, wie sich die Gesamt-Absorptionskurve "
        "des Blutes (grün) zwischen den theoretischen Kurven von sauerstoffarmem Hämoglobin (blau, entspricht geringer $O_2$-Sättigung) und sauerstoffreichem Hämoglobin (rot, entspricht hoher $O_2$-Sättigung) hin- und herverschiebt.\n\n"
        "Achten Sie auf die Schnittpunkte mit den Wellenlängen **660 nm** und **940 nm**: Ein Pulsoximeter sendet Licht in genau diesen Farben (Wellenlängen) aus, der verbaute Sensor misst die Absorption dieses Lichts."
        "Aus diesem Verhältnis der Lichtintensität, die beim Detektor ankommt (rechte Grafik), bestimmt der Detektor die Sauerstoffsättigung."
    )

    # Steuerung über einen prominenten Slider oben
    spo2 = st.slider("Eingestellte Sauerstoffsättigung ($SpO_2$ in %):", min_value=70, max_value=100, value=95, step=1)

    # Layout für die beiden Graphen nebeneinander
    col1, col2 = st.columns(2)

    # --- OPTIMIERTE ANPASSUNG IN 10-NM-INTERVALLEN ---
    wellenlaengen_stuetz = np.arange(600, 1010, 10)
    
    hb_stuetz = np.array([
        10.0, 7.8, 6.0, 4.8, 4.0, 3.4, 2.9, 2.5, 2.2, 1.9,  
        1.7, 1.5, 1.35, 1.22, 1.12, 1.18, 1.4, 1.45, 1.25, 1.05, 
        0.85, 0.73, 0.67, 0.65, 0.64, 0.63, 0.62, 0.62, 0.63, 0.65, 
        0.68, 0.7, 0.72, 0.72, 0.71, 0.65, 0.52, 0.38, 0.22, 0.12, 
        0.06 
    ])
    
    hbo2_stuetz = np.array([
        2.5, 1.7, 1.1, 0.75, 0.52, 0.4, 0.32, 0.27, 0.25, 0.26, 
        0.28, 0.31, 0.35, 0.4, 0.46, 0.53, 0.61, 0.69, 0.76, 0.83, 
        0.9, 0.96, 1.01, 1.05, 1.1, 1.13, 1.16, 1.18, 1.2, 1.21, 
        1.22, 1.22, 1.21, 1.2, 1.18, 1.16, 1.13, 1.09, 1.04, 0.98, 
        0.92 
    ])

    wellenlaengen = np.linspace(600, 1000, 300)
    abs_hb = np.interp(wellenlaengen, wellenlaengen_stuetz, hb_stuetz)
    abs_hbo2 = np.interp(wellenlaengen, wellenlaengen_stuetz, hbo2_stuetz)

    fraktion_hbo2 = spo2 / 100.0
    fraktion_hb = 1.0 - fraktion_hbo2
    abs_gesamt = (fraktion_hb * abs_hb) + (fraktion_hbo2 * abs_hbo2)

    idx_660 = np.abs(wellenlaengen - 660).argmin()
    idx_940 = np.abs(wellenlaengen - 940).argmin()
    
    abs_at_660 = abs_gesamt[idx_660]
    abs_at_940 = abs_gesamt[idx_940]

    with col1:
        st.subheader("📈 Optisches Spektrum & Absorption")
        fig_spec = go.Figure()
        
        fig_spec.add_trace(go.Scatter(x=wellenlaengen, y=abs_hb, mode='lines', name='Deoxygeniertes Hb', line=dict(color='#1976D2', width=2)))
        fig_spec.add_trace(go.Scatter(x=wellenlaengen, y=abs_hbo2, mode='lines', name='Oxygeniertes HbO₂', line=dict(color='#D32F2F', width=2)))
        fig_spec.add_trace(go.Scatter(x=wellenlaengen, y=abs_gesamt, mode='lines', name='Gesamt-Blutabsorption', line=dict(color='#2E7D32', width=3)))
        
        fig_spec.add_vline(x=660, line_width=2, line_dash="dash", line_color="red")
        fig_spec.add_vline(x=940, line_width=2, line_dash="dash", line_color="purple")
        
        fig_spec.add_trace(go.Scatter(x=[660], y=[abs_at_660], mode='markers', marker=dict(color='red', size=10, symbol='circle'), name='Messpunkt 660 nm'))
        fig_spec.add_trace(go.Scatter(x=[940], y=[abs_at_940], mode='markers', marker=dict(color='purple', size=10, symbol='circle'), name='Messpunkt 940 nm'))
        
        fig_spec.update_layout(
            xaxis_title="Wellenlänge (nm)",
            yaxis_title="Molarer Extinktionskoeffizient (1/(cm*mM))",
            xaxis=dict(range=[600, 1000], tickvals=[600, 650, 700, 750, 800, 850, 900, 950, 1000]),
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
        
        intensitaet_base_660 = 2.5 * np.exp(-abs_at_660 * 0.4)
        intensitaet_base_940 = 2.5 * np.exp(-abs_at_940 * 0.4)
        
        sig_660 = intensitaet_base_660 - (0.05 * intensitaet_base_660 * puls)
        sig_940 = intensitaet_base_940 - (0.05 * intensitaet_base_940 * puls)
        
        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(x=t, y=sig_660, mode='lines', name='Detektorsignal Rot (660 nm)', line=dict(color='red', width=2)))
        fig_sig.add_trace(go.Scatter(x=t, y=sig_940, mode='lines', name='Detektorsignal IR (940 nm)', line=dict(color='purple', width=2)))
        
        fig_sig.update_layout(
            # xaxis_title="Zeit (Sekunden)",
            yaxis_title="Empfangene Lichtintensität (V)",
            yaxis=dict(range=[0, 2.5]),
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
    
   # --- ÜBERARBEITETER RECHENWEG ---
    with st.expander("Der mathematische Rechenweg"):
        st.markdown(
            r"""
            ### 1. Das physikalische Messprinzip
            Ein Pulsoximeter bestimmt die Sauerstoffsättigung, indem es die Lichtschwächung bei zwei ganz spezifischen Wellenlängen vergleicht. 
            Dazu besitzt der Sensor zwei Leuchtdioden: **Rot (660 nm)** und **Infrarot (940 nm)**. 

            Da deoxygeniertes Hämoglobin ($Hb$) und oxygeniertes Hämoglobin ($HbO_2$) unterschiedliche Absorptionsverhalten zeigen, 
            verschiebt sich die grüne Gesamtabsorptionskurve je nach Zusammensetzung des Blutes.

            ### 2. Die mathematische Verhältnisbildung ($R$-Wert)
            Um unabhängig von der individuellen Fingerdicke, der Hautfarbe oder der LED-Helligkeit zu messen, normiert das Gerät die Signale und berechnet das Verhältnis der Absorptionen zueinander:
            """
        )
        
        # Die mathematische Formel als separater F-String mit doppelten Backslashes für LaTeX
        st.write(f"$$R = \\frac{{\\text{{Absorption bei }} 660\\,\\text{{nm}}}}{{\\text{{Absorption bei }} 940\\,\\text{{nm}}}} = \\frac{{{abs_at_660:.3f}}}{{{abs_at_940:.3f}}} = {r_wert_berechnet:.3f}$$")
        
        st.markdown(
            r"""
            ### 3. Didaktische Fallbeispiele zur Veranschaulichung

            * **Fall A: Hohe Sättigung (z.B. 100% $SpO_2$)**
              - Das Blut besteht fast ausschließlich aus $HbO_2$ (rote Kurve).
              - Schauen Sie auf die Schnittpunkte: Bei $660\,\text{nm}$ ist die rote Kurve auf einem physiologischen Minimum (~0.25). Bei $940\,\text{nm}$ im Infrarotbereich absorbiert sie deutlich stärker (~1.2).
              - Der Zähler ist also klein, der Nenner groß $\rightarrow$ **Der $R$-Wert wird klein ($\approx 0.4 - 0.5$).**
              
            * **Fall B: Schlechte Sättigung (z.B. 70% $SpO_2$)**
              - Das ungesättigte $Hb$ (blaue Kurve) gewinnt die Oberhand.
              - Bei $660\,\text{nm}$ schießt die blaue Kurve dramatisch nach oben ($\approx 2.5$). Im Infrarotbereich bei $940\,\text{nm}$ fällt sie hingegen weit ab ($\approx 0.5$).
              - Der Zähler wird riesig, der Nenner klein $\rightarrow$ **Der $R$-Wert steigt stark an ($\approx 1.5 - 2.0$).**

            ### 4. Von der Kurve zum Prozentwert
            Das Medizintechnikgerät berechnet im ersten Schritt also ausschließlich diesen dimensionslosen **$R$-Wert**. Im Mikrocontroller des Sensors ist eine empirisch ermittelte Kalibrationskurve hinterlegt. Diese ordnet jedem berechneten Verhältniswert die exakte Sättigung in Prozent zu.
            
            * **Grobe Faustformel für die Praxis:** $SpO_2 \approx 110 - 25 \cdot R$
            """
        )
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    spo2_applet()
