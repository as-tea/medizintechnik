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

# -----------------------------------------------------------------------------
# STYLING DEFINITION FÜR DAS SORTABLE-WIDGET
# -----------------------------------------------------------------------------
# Dunkelblaues Design mit weißer, fetter Schrift direkt als CSS-Style übergeben
custom_sortable_style = """
    ul {
        padding: 0 !important;
    }
    li {
        background-color: #1E3A8A !important;  /* Dunkelblau */
        color: #FFFFFF !important;              /* Weißer Text */
        font-weight: bold !important;          /* Fette Schrift */
        font-size: 1.05rem !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        margin-bottom: 8px !important;
        border: 1px solid #3B82F6 !important;   /* Hellerer blauer Rand */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        list-style-type: none !important;
    }
    li:hover {
        background-color: #2563EB !important;  /* Etwas helleres Blau beim Drüberfahren */
        cursor: grab !important;
    }
"""

# Custom CSS für Badges
st.markdown("""
    <style>
    .phase-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.9rem;
        margin-bottom: 8px;
        color: #FFFFFF !important;
    }
    .phase-1 { background-color: #1E3A8A; }  /* Beschaffung: Dunkelblau */
    .phase-2 { background-color: #0D9488; }  /* Freigabe: Türkis/Dunkelgrün */
    .phase-3 { background-color: #D97706; }  /* Betrieb: Bernstein/Dunkelgelb */
    .phase-4 { background-color: #DC2626; }  /* Entsorgung: Rot */
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATEN & KORREKTE REIHENFOLGEN
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

def format_item(key):
    return f"[{key}]  {TASKS[key]}"

CORRECT_ORDER_1 = ["D", "G", "E", "A", "I", "J", "C", "H", "F", "B"]
CORRECT_ITEMS_1 = [format_item(k) for k in CORRECT_ORDER_1]

CORRECT_ORDER_2 = ["D", "G", "E", "A", "I", "J", "H", "C", "F", "B"]
CORRECT_ITEMS_2 = [format_item(k) for k in CORRECT_ORDER_2]

PHASES = [
    ("Phase 1: Beschaffung & Zulassung", "phase-1", ["D", "G", "E"]),
    ("Phase 2: Qualifizierung & Freigabe", "phase-2", ["A", "I"]),
    ("Phase 3: Regulärer Betrieb & Instandhaltung", "phase-3", ["J", "C / H", "F"]),
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

st.info("💡 **Bedienung:** Ziehen Sie die dunkelblauen Kacheln per **Drag & Drop** an die richtige Position und klicken Sie anschließend auf **Reihenfolge prüfen**.")

st.divider()

# -----------------------------------------------------------------------------
# DRAG & DROP BEREICH
# -----------------------------------------------------------------------------
if "current_items" not in st.session_state:
    st.session_state.current_items = [format_item(k) for k in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]]

st.subheader("🛠️ Sortieren Sie die Schritte:")

# Hier übergeben wir den Style direkt an das sort_items Widget!
sorted_items = sort_items(
    st.session_state.current_items, 
    direction="vertical",
    custom_style=custom_sortable_style
)

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
    match_1 = sum(1 for a, b in zip(sorted_items, CORRECT_ITEMS_1) if a == b)
    match_2 = sum(1 for a, b in zip(sorted_items, CORRECT_ITEMS_2) if a == b)
    
    correct_count = max(match_1, match_2)
    total_count = len(CORRECT_ITEMS_1)
    
    if correct_count == total_count:
        st.balloons()
        st.success("🎉 **Perfekt!** Sie haben alle 10 Schritte in die exakt richtige Reihenfolge gebracht!")
    else:
        st.warning(f"🎯 **Ergebnis:** {correct_count} von {total_count} Schritten stehen bereits an der richtigen Position.")
        st.caption("Tipp: STK und MTK sind in ihrer Reihenfolge untereinander austauschbar. Überlegen Sie, welche Schritte zwingend VOR der ersten Anwendung am Patienten stattfinden müssen.")

# -----------------------------------------------------------------------------
# MUSTERLÖSUNG
# -----------------------------------------------------------------------------
if show_solution:
    st.subheader("📖 Musterlösung & Lebenszyklus-Phasen")
    
    for phase_name, badge_class, keys in PHASES:
        st.markdown(f'<span class="phase-badge {badge_class}">{phase_name}</span>', unsafe_allow_html=True)
        for k in keys:
            if " / " in k:
                st.markdown(f"- **[C / H]** {TASKS['C']} *sowie* {TASKS['H']} *(Reihenfolge untereinander austauschbar)*")
            else:
                st.markdown(f"- **[{k}]** {TASKS[k]}")
        st.write("")
