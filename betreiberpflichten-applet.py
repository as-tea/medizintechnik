import streamlit as st
from streamlit_sortables import sort_items

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Betreiberpflichten-Sortierer",
    page_icon="📋",
    layout="centered"
)

# Custom CSS für ansprechendes Karten-Design
st.markdown("""
    <style>
    .phase-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85rem;
        margin-bottom: 8px;
    }
    .phase-1 { background-color: #e3f2fd; color: #1565c0; }
    .phase-2 { background-color: #e8f5e9; color: #2e7d32; }
    .phase-3 { background-color: #fff8e1; color: #f57f17; }
    .phase-4 { background-color: #ffebee; color: #c62828; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATEN & KORREKTE REIHENFOLGE (DGEAIJCHFB)
# -----------------------------------------------------------------------------
TASKS = {
    "A": "Schulung des Anwenderpersonals (nach MPBetreibV)",
    "B": "Außerbetriebnahme/Entsorgung",
    "C": "Durchführung der Sicherheitstechnischen Kontrollen (STK)",
    "D": "Entscheidung zur Anschaffung",
    "E": "Anlegen der Gerätedatei (Dokumentationspflicht)",
    "F": "Fehleranalyse und ggf. Meldung (bei Vorkommnissen)",
    "G": "Überprüfung des Konformitätszeichens (CE-Kennzeichnung)",
    "H": "Durchführung der messtechnischen Kontrollen (MTK)",
    "I": "Übergabe und Inbetriebnahme auf der Station",
    "J": "Regelmäßige Sichtprüfung und Funktionskontrolle durch den Anwender"
}

CORRECT_ORDER_KEYS = ["D", "G", "E", "A", "I", "J", "C", "H", "F", "B"]

# Formattierte Texte für die Sortierliste
def format_item(key):
    return f"[{key}]  {TASKS[key]}"

CORRECT_ITEMS = [format_item(k) for k in CORRECT_ORDER_KEYS]

# Phasen-Zuordnung für die Lösungsübersicht
PHASES = [
    ("Phase 1: Beschaffung & Zulassung", "phase-1", ["D", "G", "E"]),
    ("Phase 2: Qualifizierung & Freigabe", "phase-2", ["A", "I"]),
    ("Phase 3: Regulärer Betrieb & Instandhaltung", "phase-3", ["J", "C", "H", "F"]),
    ("Phase 4: Außerbetriebnahme", "phase-4", ["B"])
]

# -----------------------------------------------------------------------------
# HEADER & EINLEITUNG
# -----------------------------------------------------------------------------
st.title("📋 Do-it-yourself: Betreiberpflichten")

st.markdown("""
Medizinprodukte unterliegen über ihren gesamten Lebenszyklus hinweg strengen rechtlichen Betreiberpflichten (gemäß MPBetreibV).
Bringen Sie die folgenden zehn Schritte von der ersten Anschaffung bis zur Entsorgung in die **korrekte logische und zeitliche Reihenfolge**!
""")

st.info("💡 **Bedienung:** Ziehen Sie die Elemente per **Drag & Drop** an die richtige Position und klicken Sie anschließend auf **Reihenfolge prüfen**.")

st.divider()

# -----------------------------------------------------------------------------
# DRAG & DROP BEREICH
# -----------------------------------------------------------------------------
# Start-Reihenfolge (Ungeordnet, z. B. A bis J)
if "current_items" not in st.session_state:
    st.session_state.current_items = [format_item(k) for k in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]]

st.subheader("🛠️ Sortieren Sie die Schritte:")

# Sortable Widget
sorted_items = sort_items(st.session_state.current_items, direction="vertical")

st.divider()

# -----------------------------------------------------------------------------
# AUSWERTUNG
# -----------------------------------------------------------------------------
col_btn1, col_btn2 = st.columns([1, 1])

with col_btn1:
    check_clicked = st.button("🔍 Reihenfolge prüfen", type="primary", use_container_width=True)

with col_btn2:
    show_solution = st.checkbox("💡 Musterlösung anzeigen", value=False)

if check_clicked:
    # Berechne wie viele Elemente an der exakt richtigen Stelle stehen
    correct_count = sum(1 for a, b in zip(sorted_items, CORRECT_ITEMS) if a == b)
    total_count = len(CORRECT_ITEMS)
    
    if correct_count == total_count:
        st.balloons()
        st.success("🎉 **Perfekt!** Sie haben alle 10 Schritte in die exakt richtige Reihenfolge gebracht!")
    else:
        st.warning(f"🎯 **Ergebnis:** {correct_count} von {total_count} Schritten stehen bereits an der richtigen Position.")
        st.caption("Tipp: Überlegen Sie, welche Schritte zwingend VOR der ersten Anwendung am Patienten stattfinden müssen.")

# -----------------------------------------------------------------------------
# MUSTERLÖSUNG & LEBENSZYKLUS-PHASEN
# -----------------------------------------------------------------------------
if show_solution:
    st.subheader("📖 Musterlösung & Lebenszyklus-Phasen")
    
    for phase_name, badge_class, keys in PHASES:
        st.markdown(f'<span class="phase-badge {badge_class}">{phase_name}</span>', unsafe_allow_html=True)
        for k in keys:
            st.markdown(f"- **[{k}]** {TASKS[k]}")
        st.write("")
