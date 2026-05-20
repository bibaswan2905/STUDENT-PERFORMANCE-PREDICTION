import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import os
import io
from datetime import datetime

from data_creation import create_student_data, STUDENT_PASSWORDS, STUDENT_NAMES
from prediction_model import (
    load_data, build_features, train_model,
    predict_student, get_recommendations,
    YOUTUBE_RECOMMENDATIONS, BOOK_RECOMMENDATIONS
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
*, html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #060b14; }
section[data-testid="stSidebar"] { display: none; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1200px !important; }

/* ── METRIC CARDS ── */
.metric-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
.metric-card {
    background: linear-gradient(145deg, #0d1829 0%, #0a1220 100%);
    border: 1px solid #1e3a5f; border-radius: 14px; padding: 20px 22px; position: relative; overflow: hidden;
}
.metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; }
.mc-blue::before   { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.mc-purple::before { background: linear-gradient(90deg, #7c3aed, #a78bfa); }
.mc-green::before  { background: linear-gradient(90deg, #059669, #34d399); }
.mc-orange::before { background: linear-gradient(90deg, #d97706, #fbbf24); }
.metric-icon  { font-size: 1.4rem; margin-bottom: 8px; display: block; }
.metric-value { font-size: 1.9rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; line-height: 1; margin-bottom: 4px; }
.mv-blue   { color: #60a5fa; } .mv-purple { color: #a78bfa; }
.mv-green  { color: #34d399; } .mv-orange { color: #fbbf24; }
.metric-label { color: #475569; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }

/* ── SUBJECT TABLE ── */
.subject-table { background: #0a1220; border: 1px solid #1e3a5f; border-radius: 14px; overflow: hidden; margin-bottom: 24px; }
.subject-table-header {
    background: #0d1829; padding: 12px 24px; border-bottom: 1px solid #1e3a5f;
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.5px; color: #475569; font-weight: 600;
    display: grid; grid-template-columns: 2fr 1.2fr 0.8fr 0.8fr 1fr; gap: 8px;
}
.subject-row {
    padding: 12px 24px; border-bottom: 1px solid #0d1829;
    display: grid; grid-template-columns: 2fr 1.2fr 0.8fr 0.8fr 1fr; gap: 8px; align-items: center;
}
.subject-row:last-child { border-bottom: none; }
.subject-row:hover { background: rgba(96,165,250,0.04); }
.subj-name { color: #cbd5e1; font-size: 0.88rem; font-weight: 500; }
.subj-cluster-badge { display: inline-block; padding: 3px 10px; border-radius: 6px; font-size: 0.7rem; font-weight: 600; }
.bc-prog   { background: rgba(59,130,246,0.12); color: #60a5fa; border: 1px solid rgba(59,130,246,0.2); }
.bc-theory { background: rgba(139,92,246,0.12); color: #a78bfa; border: 1px solid rgba(139,92,246,0.2); }
.bc-math   { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.2); }
.subj-score { font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; font-weight: 600; color: #e2e8f0; }
.grade-badge { display: inline-block; padding: 3px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.status-pass { color: #34d399; font-size: 0.78rem; font-weight: 600; }
.status-fail { color: #f87171; font-size: 0.78rem; font-weight: 600; }

/* ── SECTION TITLE ── */
.section-title {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 2px; color: #475569; font-weight: 600;
    margin: 28px 0 14px 0; padding-bottom: 10px; border-bottom: 1px solid #0d1829;
}

/* ── CLUSTER CARD ── */
.cluster-card { border-radius: 14px; padding: 24px 28px; margin-bottom: 16px; }
.cc-prog   { background: linear-gradient(135deg, #0d1f3c, #0a1828); border: 1px solid rgba(59,130,246,0.25); }
.cc-theory { background: linear-gradient(135deg, #150d2e, #100a22); border: 1px solid rgba(139,92,246,0.25); }
.cc-math   { background: linear-gradient(135deg, #082819, #061d12); border: 1px solid rgba(52,211,153,0.25); }
.cc-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px; }
.cc-score  { font-size: 1.8rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
.cc-score-prog   { color: #60a5fa; } .cc-score-theory { color: #a78bfa; } .cc-score-math { color: #34d399; }
.pb-bg { background: rgba(255,255,255,0.06); border-radius: 999px; height: 5px; margin: 8px 0 14px; overflow: hidden; }
.pb-fill { height: 100%; border-radius: 999px; }
.pb-prog { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.pb-theory { background: linear-gradient(90deg, #7c3aed, #a78bfa); }
.pb-math { background: linear-gradient(90deg, #059669, #34d399); }
.cc-insight { color: #94a3b8; font-size: 0.82rem; line-height: 1.65; margin-bottom: 12px; }
.cc-warn { background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.2); border-radius: 8px; padding: 9px 14px; color: #fbbf24; font-size: 0.78rem; margin-bottom: 12px; }
.cc-ok   { background: rgba(52,211,153,0.08); border: 1px solid rgba(52,211,153,0.2); border-radius: 8px; padding: 9px 14px; color: #34d399; font-size: 0.78rem; margin-bottom: 12px; }

/* ── REC ITEMS ── */
.rec-item { background: rgba(255,255,255,0.03); border: 1px solid #1e3a5f; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; }
.rec-item-title { color: #cbd5e1; font-size: 0.82rem; font-weight: 500; margin-bottom: 4px; }
.rec-item-meta  { color: #475569; font-size: 0.76rem; }
.rec-item-link  { color: #60a5fa; font-size: 0.76rem; word-break: break-all; }

/* ── ALERTS ── */
.alert-warn { background: rgba(251,191,36,0.07); border: 1px solid rgba(251,191,36,0.25); border-radius: 10px; padding: 12px 18px; color: #fbbf24; font-size: 0.85rem; margin: 10px 0; }
.alert-ok   { background: rgba(52,211,153,0.07); border: 1px solid rgba(52,211,153,0.25); border-radius: 10px; padding: 12px 18px; color: #34d399; font-size: 0.85rem; margin: 10px 0; }
.error-msg  { background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.25); border-radius: 10px; padding: 12px 18px; color: #f87171; font-size: 0.85rem; margin-top: 12px; text-align: center; }

/* ── BUTTONS ── */
div.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important; color: white !important;
    border: none !important; border-radius: 10px !important; font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important; padding: 10px 24px !important;
    font-size: 0.9rem !important; box-shadow: 0 4px 16px rgba(59,130,246,0.25) !important;
}
div.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 20px rgba(59,130,246,0.4) !important; }

/* ── INPUTS ── */
.stTextInput input, .stNumberInput input {
    background: #0d1829 !important; border: 1px solid #1e3a5f !important;
    color: #e2e8f0 !important; border-radius: 10px !important;
}
.stTextInput label, .stNumberInput label {
    color: #64748b !important; font-size: 0.8rem !important;
    font-weight: 500 !important; text-transform: uppercase !important; letter-spacing: 1px !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] { background: #0a1220 !important; border-radius: 10px !important; padding: 4px !important; border: 1px solid #1e3a5f !important; gap: 4px !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #475569 !important; border-radius: 8px !important; font-weight: 600 !important; font-size: 0.85rem !important; padding: 8px 20px !important; }
.stTabs [aria-selected="true"] { background: #1d4ed8 !important; color: white !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 24px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA + MODEL INIT
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def init_data():
    if not os.path.exists("output/Student_Performance_Prediction.xlsx"):
        create_student_data()
    df1, df2, df3 = load_data()
    merged = build_features(df1, df2, df3)
    return df1, df2, df3, merged

@st.cache_resource
def init_models(merged):
    if os.path.exists("output/models.pkl"):
        with open("output/models.pkl", "rb") as f:
            return pickle.load(f)
    return train_model(merged)

df1, df2, df3, merged = init_data()
models = init_models(merged)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_grade(score):
    if score >= 90: return "O",   "#34d399", "rgba(52,211,153,0.12)",  "rgba(52,211,153,0.35)"
    if score >= 75: return "A+",  "#60a5fa", "rgba(96,165,250,0.12)",  "rgba(96,165,250,0.35)"
    if score >= 60: return "A",   "#818cf8", "rgba(129,140,248,0.12)", "rgba(129,140,248,0.35)"
    if score >= 50: return "B",   "#fbbf24", "rgba(251,191,36,0.12)",  "rgba(251,191,36,0.35)"
    if score >= 40: return "C",   "#f97316", "rgba(249,115,22,0.12)",  "rgba(249,115,22,0.35)"
    return             "F",   "#f87171", "rgba(248,113,113,0.12)", "rgba(248,113,113,0.35)"

def att_status(a):
    if a >= 85: return "Excellent", "#34d399"
    if a >= 75: return "Good",      "#60a5fa"
    if a >= 65: return "Average",   "#fbbf24"
    return             "Critical",  "#f87171"

CHART_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94a3b8', family='Inter'), margin=dict(l=16, r=16, t=44, b=16)
)

CLUSTER_META = {
    "Programming": {
        "icon": "🖥️", "color": "#60a5fa", "cc": "cc-prog",
        "score_cls": "cc-score-prog", "pb": "pb-prog",
        "subjects": {"Sem1": "C Programming", "Sem2": "Data Structures", "Sem3": "Algorithms"},
        "cols": [("C_Programming","C Programming"),("Data_Structures","Data Structures"),("Algorithms","Algorithms")]
    },
    "Theory": {
        "icon": "📚", "color": "#a78bfa", "cc": "cc-theory",
        "score_cls": "cc-score-theory", "pb": "pb-theory",
        "subjects": {"Sem1": "Digital Logic", "Sem2": "Computer Org.", "Sem3": "Operating Systems"},
        "cols": [("Digital_Logic","Digital Logic"),("Computer_Organization","Comp. Org."),("Operating_Systems","OS")]
    },
    "Math_Logic": {
        "icon": "📐", "color": "#34d399", "cc": "cc-math",
        "score_cls": "cc-score-math", "pb": "pb-math",
        "subjects": {"Sem1": "Maths I", "Sem2": "Discrete Maths", "Sem3": "DBMS"},
        "cols": [("Maths_I","Maths I"),("Discrete_Maths","Discrete Maths"),("DBMS","DBMS")]
    }
}

SUBJECT_ROWS = [
    ("C Programming",     "C_Programming",         "Programming", "bc-prog",   "Sem 1"),
    ("Maths I",           "Maths_I",               "Math/Logic",  "bc-math",   "Sem 1"),
    ("Digital Logic",     "Digital_Logic",         "Theory",      "bc-theory", "Sem 1"),
    ("Data Structures",   "Data_Structures",       "Programming", "bc-prog",   "Sem 2"),
    ("Discrete Maths",    "Discrete_Maths",        "Math/Logic",  "bc-math",   "Sem 2"),
    ("Computer Org.",     "Computer_Organization", "Theory",      "bc-theory", "Sem 2"),
    ("Algorithms",        "Algorithms",            "Programming", "bc-prog",   "Sem 3"),
    ("Operating Systems", "Operating_Systems",     "Theory",      "bc-theory", "Sem 3"),
    ("DBMS",              "DBMS",                  "Math/Logic",  "bc-math",   "Sem 3"),
]


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def chart_attendance(student):
    sems = ["Semester 1", "Semester 2", "Semester 3"]
    vals = [float(student["Sem1_Attendance"].values[0]),
            float(student["Sem2_Attendance"].values[0]),
            float(student["Sem3_Attendance"].values[0])]
    colors = ['#34d399' if v >= 75 else '#fbbf24' if v >= 65 else '#f87171' for v in vals]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sems, y=vals, marker=dict(color=colors, line=dict(width=0)),
                         text=[f"{v:.0f}%" for v in vals], textposition='inside',
                         textfont=dict(size=14, color='white', family='JetBrains Mono')))
    fig.add_hline(y=75, line_dash="dot", line_color="#fbbf24", line_width=1.5,
                  annotation_text="  75% Min", annotation_font=dict(color="#fbbf24", size=11))
    fig.update_layout(title=dict(text="Attendance Per Semester", font=dict(size=13, color='#94a3b8')),
                      yaxis=dict(range=[0, 110], gridcolor='#0d1829'), xaxis=dict(gridcolor='rgba(0,0,0,0)'),
                      **CHART_BASE)
    return fig

def chart_cluster_pie(student):
    labels = ["Programming", "Theory", "Math / Logic"]
    values = [float(student["Programming_Avg"].values[0]),
              float(student["Theory_Avg"].values[0]),
              float(student["Math_Logic_Avg"].values[0])]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=["#3b82f6","#7c3aed","#059669"], line=dict(color='#060b14', width=3)),
        textinfo='label+percent', textfont=dict(size=11, color='white'),
    ))
    fig.add_annotation(text=f"Overall<br><b>{np.mean(values):.1f}</b>",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=13, color='#94a3b8', family='Inter'))
    fig.update_layout(title=dict(text="Cluster Score Distribution", font=dict(size=13, color='#94a3b8')),
                      showlegend=False, **CHART_BASE)
    return fig

def chart_performance_trend(student):
    sems = ["Semester 1", "Semester 2", "Semester 3"]
    avgs = [float(student["Sem1_Avg"].values[0]),
            float(student["Sem2_Avg"].values[0]),
            float(student["Sem3_Avg"].values[0])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sems, y=avgs, mode='lines+markers',
                             line=dict(color='#3b82f6', width=3),
                             marker=dict(size=9, color='#60a5fa', line=dict(color='#060b14', width=2)),
                             fill='tozeroy', fillcolor='rgba(59,130,246,0.08)', name='Overall'))
    for cluster, meta in CLUSTER_META.items():
        cvals = [float(student[f"{cluster}_Sem1"].values[0]),
                 float(student[f"{cluster}_Sem2"].values[0]),
                 float(student[f"{cluster}_Sem3"].values[0])]
        fig.add_trace(go.Scatter(x=sems, y=cvals, mode='lines+markers',
                                 line=dict(color=meta["color"], width=1.5, dash='dot'),
                                 marker=dict(size=6, color=meta["color"]),
                                 name=cluster.replace("_","/"), opacity=0.7))
    fig.update_layout(title=dict(text="Performance Trend Across Semesters", font=dict(size=13, color='#94a3b8')),
                      yaxis=dict(range=[0,100], gridcolor='#0d1829'),
                      xaxis=dict(gridcolor='rgba(0,0,0,0)'),
                      legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1e3a5f', borderwidth=1,
                                  font=dict(size=10, color='#94a3b8')),
                      **CHART_BASE)
    return fig

def chart_predicted(student, predictions):
    sems_all = ["Semester 1", "Semester 2", "Semester 3", "Semester 4\n(Predicted)"]
    pred_avg = float(np.mean(list(predictions.values())))
    avgs = [float(student["Sem1_Avg"].values[0]),
            float(student["Sem2_Avg"].values[0]),
            float(student["Sem3_Avg"].values[0]), pred_avg]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sems_all[:3], y=avgs[:3], mode='lines+markers',
                             line=dict(color='#60a5fa', width=3),
                             marker=dict(size=9, color='#60a5fa', line=dict(color='#060b14', width=2)),
                             fill='tozeroy', fillcolor='rgba(96,165,250,0.07)', name='Actual'))
    fig.add_trace(go.Scatter(x=[sems_all[2], sems_all[3]], y=[avgs[2], pred_avg],
                             mode='lines+markers',
                             line=dict(color='#a78bfa', width=3, dash='dash'),
                             marker=dict(size=11, color='#a78bfa', symbol='diamond',
                                         line=dict(color='#060b14', width=2)),
                             name='Predicted'))
    fig.add_annotation(x=sems_all[3], y=pred_avg, text=f"  {pred_avg:.1f}",
                       showarrow=False, font=dict(size=13, color='#a78bfa', family='JetBrains Mono'))
    fig.update_layout(title=dict(text="Predicted Semester 4 Performance", font=dict(size=13, color='#94a3b8')),
                      yaxis=dict(range=[0,100], gridcolor='#0d1829'),
                      xaxis=dict(gridcolor='rgba(0,0,0,0)'),
                      legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1e3a5f', borderwidth=1,
                                  font=dict(size=10, color='#94a3b8')),
                      **CHART_BASE)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PDF GENERATOR  — rich elaborated report matching Page 3
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf(student, predictions, recommendations):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                     TableStyle, HRFlowable, PageBreak, KeepTogether)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

    buf      = io.BytesIO()
    PW       = A4[0]
    PH       = A4[1]
    LM = RM  = 1.8*cm
    W        = PW - LM - RM
    doc      = SimpleDocTemplate(buf, pagesize=A4,
                                  leftMargin=LM, rightMargin=RM,
                                  topMargin=1.8*cm, bottomMargin=1.8*cm)

    # ── colour palette ───────────────────────────────────────────────────────
    C = {
        "navy":    colors.HexColor("#0f172a"),
        "blue":    colors.HexColor("#1d4ed8"),
        "blue_lt": colors.HexColor("#3b82f6"),
        "blue_bg": colors.HexColor("#eff6ff"),
        "purple":  colors.HexColor("#7c3aed"),
        "pur_lt":  colors.HexColor("#a78bfa"),
        "pur_bg":  colors.HexColor("#f5f3ff"),
        "green":   colors.HexColor("#059669"),
        "grn_lt":  colors.HexColor("#34d399"),
        "grn_bg":  colors.HexColor("#ecfdf5"),
        "orange":  colors.HexColor("#d97706"),
        "org_bg":  colors.HexColor("#fffbeb"),
        "red":     colors.HexColor("#dc2626"),
        "red_bg":  colors.HexColor("#fef2f2"),
        "grey":    colors.HexColor("#64748b"),
        "grey_lt": colors.HexColor("#94a3b8"),
        "bg1":     colors.HexColor("#f8fafc"),
        "bg2":     colors.HexColor("#f1f5f9"),
        "border":  colors.HexColor("#e2e8f0"),
        "text":    colors.HexColor("#1e293b"),
        "text2":   colors.HexColor("#334155"),
        "white":   colors.white,
    }

    CLUSTER_COLORS = {
        "Programming": (C["blue"],   C["blue_lt"],  C["blue_bg"]),
        "Theory":      (C["purple"], C["pur_lt"],   C["pur_bg"]),
        "Math_Logic":  (C["green"],  C["grn_lt"],   C["grn_bg"]),
    }

    # ── styles ───────────────────────────────────────────────────────────────
    def S(n, **k): return ParagraphStyle(n, **k)
    sty = {
        "title":   S("tt", fontName="Helvetica-Bold",   fontSize=22, textColor=C["blue"],   spaceAfter=3,  leading=28),
        "title2":  S("t2", fontName="Helvetica-Bold",   fontSize=16, textColor=C["blue"],   spaceAfter=3,  leading=20),
        "sub":     S("su", fontName="Helvetica",        fontSize=10, textColor=C["grey"],   spaceAfter=14, leading=14),
        "h1":      S("h1", fontName="Helvetica-Bold",   fontSize=13, textColor=C["text"],   spaceBefore=18, spaceAfter=7,  leading=18),
        "h2":      S("h2", fontName="Helvetica-Bold",   fontSize=11, textColor=C["text2"],  spaceBefore=12, spaceAfter=4,  leading=15),
        "h3":      S("h3", fontName="Helvetica-Bold",   fontSize=10, textColor=C["grey"],   spaceBefore=8,  spaceAfter=3,  leading=14),
        "body":    S("b",  fontName="Helvetica",        fontSize=9,  textColor=C["grey"],   leading=14,    spaceAfter=4),
        "body2":   S("b2", fontName="Helvetica",        fontSize=9,  textColor=C["text2"],  leading=14),
        "italic":  S("it", fontName="Helvetica-Oblique",fontSize=9,  textColor=C["text2"],  leading=14,    spaceAfter=6),
        "bold9":   S("b9", fontName="Helvetica-Bold",   fontSize=9,  textColor=C["text2"],  leading=14),
        "warn":    S("wn", fontName="Helvetica-Bold",   fontSize=9,  textColor=C["orange"], leading=14,    spaceAfter=5),
        "ok":      S("ok", fontName="Helvetica-Bold",   fontSize=9,  textColor=C["green"],  leading=14,    spaceAfter=5),
        "link":    S("lk", fontName="Helvetica",        fontSize=8,  textColor=C["blue_lt"],leading=12,    spaceAfter=2),
        "foot":    S("ft", fontName="Helvetica",        fontSize=7.5,textColor=C["grey_lt"],alignment=TA_CENTER),
        "badge_b": S("bb", fontName="Helvetica-Bold",   fontSize=8,  textColor=C["blue"],   leading=11),
        "badge_p": S("bp", fontName="Helvetica-Bold",   fontSize=8,  textColor=C["purple"], leading=11),
        "badge_g": S("bg", fontName="Helvetica-Bold",   fontSize=8,  textColor=C["green"],  leading=11),
    }

    # ── helpers ───────────────────────────────────────────────────────────────
    def HR(sp_before=6, sp_after=10, thick=0.5, col=None):
        return HRFlowable(width=W, thickness=thick, color=col or C["border"],
                          spaceBefore=sp_before, spaceAfter=sp_after)

    def score_bar_table(score, color, bar_width=80):
        """Return a mini Table with a visual score bar."""
        fill = int(bar_width * min(score, 100) / 100)
        bar_data = [[""]]
        t = Table(bar_data, colWidths=[bar_width], rowHeights=[6])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), C["bg2"]),
            ("LEFTPADDING", (0,0), (-1,-1), 0),
            ("RIGHTPADDING", (0,0), (-1,-1), 0),
            ("TOPPADDING", (0,0), (-1,-1), 0),
            ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ]))
        # overlay fill
        fill_data = [[""]]
        ft = Table(fill_data, colWidths=[fill], rowHeights=[6])
        ft.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), color),
            ("LEFTPADDING", (0,0), (-1,-1), 0),("RIGHTPADDING", (0,0), (-1,-1), 0),
            ("TOPPADDING", (0,0), (-1,-1), 0),("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ]))
        wrapper = Table([[ft, ""]], colWidths=[fill, bar_width - fill], rowHeights=[6])
        wrapper.setStyle(TableStyle([
            ("LEFTPADDING", (0,0), (-1,-1), 0),("RIGHTPADDING", (0,0), (-1,-1), 0),
            ("TOPPADDING", (0,0), (-1,-1), 0),("BOTTOMPADDING", (0,0), (-1,-1), 0),
            ("BACKGROUND", (0,0), (-1,-1), C["bg2"]),
            ("BACKGROUND", (0,0), (0,-1), color),
        ]))
        return wrapper

    # ── data ─────────────────────────────────────────────────────────────────
    name      = student["Name"].values[0]
    roll      = int(student["Roll_No"].values[0])
    sem3_avg  = float(student["Sem3_Avg"].values[0])
    att_avg   = float(student["Attendance_Avg"].values[0])
    pred_avg  = float(np.mean(list(predictions.values())))
    grade, gc, *_ = get_grade(sem3_avg)
    pred_grade, pgc, *_ = get_grade(pred_avg)
    a_stat, _  = att_status(att_avg)
    s1a = int(student["Sem1_Attendance"].values[0])
    s2a = int(student["Sem2_Attendance"].values[0])
    s3a = int(student["Sem3_Attendance"].values[0])

    story = []

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 1 — COVER / SUMMARY
    # ════════════════════════════════════════════════════════════════════════

    # Banner bar
    banner = Table([["  Student Performance Report  —  B.Tech CSE"]], colWidths=[W], rowHeights=[28])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C["navy"]),
        ("TEXTCOLOR",  (0,0), (-1,-1), C["white"]),
        ("FONTNAME",   (0,0), (-1,-1), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 11),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",(0,0), (-1,-1), 12),
        ("TOPPADDING", (0,0), (-1,-1), 0),
        ("BOTTOMPADDING",(0,0),(-1,-1), 0),
    ]))
    story.append(banner)
    story.append(Spacer(1, 14))

    story.append(Paragraph(f"Academic Report — {name}", sty["title"]))
    story.append(Paragraph(
        f"Roll No: #{roll:02d}  |  Programme: B.Tech Computer Science & Engineering  |"
        f"  Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
        sty["sub"]
    ))
    story.append(HR(thick=1.5, col=C["blue"], sp_before=0, sp_after=16))

    # ── Summary metric row (4 boxes) ────────────────────────────────────────
    def info_box(label, value, sub, bg, border_col, text_col):
        t = Table([[Paragraph(label, S("lbl", fontName="Helvetica", fontSize=7.5,
                                        textColor=C["grey"], leading=10)),
                    Paragraph(value, S("val", fontName="Helvetica-Bold", fontSize=18,
                                        textColor=text_col, leading=22)),
                    Paragraph(sub,   S("sb",  fontName="Helvetica", fontSize=8,
                                        textColor=C["grey"], leading=11))]],
                  colWidths=[W*0.24], rowHeights=[56])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), bg),
            ("BOX",           (0,0), (-1,-1), 1.5, border_col),
            ("ROUNDEDCORNERS",(0,0), (-1,-1), [4]),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("RIGHTPADDING",  (0,0), (-1,-1), 10),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]))
        return t

    boxes = Table([[
        info_box("CURRENT GRADE",    grade,            f"Sem 3 average: {sem3_avg:.1f}",  C["blue_bg"], C["blue_lt"], C["blue"]),
        info_box("PREDICTED SEM 4",  f"{pred_avg:.1f}", f"Grade {pred_grade} expected",   C["pur_bg"],  C["pur_lt"],  C["purple"]),
        info_box("ATTENDANCE",       f"{att_avg:.0f}%", a_stat,                           C["grn_bg"],  C["grn_lt"],  C["green"]),
        info_box("SEM 3 SCORE",      f"{sem3_avg:.1f}", "Overall average",                C["org_bg"],  C["orange"],  C["orange"]),
    ]], colWidths=[W*0.245]*4, rowHeights=[72])
    boxes.setStyle(TableStyle([
        ("LEFTPADDING",  (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING",   (0,0), (-1,-1), 0),
        ("BOTTOMPADDING",(0,0), (-1,-1), 0),
    ]))
    story.append(boxes)
    story.append(Spacer(1, 18))

    # ── Attendance alert ─────────────────────────────────────────────────────
    if att_avg < 75:
        att_box = Table([[
            Paragraph("⚠  ATTENDANCE WARNING", S("aw", fontName="Helvetica-Bold", fontSize=9, textColor=C["orange"], leading=12)),
            Paragraph(
                f"Your average attendance of {att_avg:.1f}% is below the mandatory 75% requirement. "
                f"Students below 75% may be barred from semester examinations. "
                f"Sem 1: {s1a}%  |  Sem 2: {s2a}%  |  Sem 3: {s3a}%",
                S("at", fontName="Helvetica", fontSize=8.5, textColor=C["orange"], leading=13)
            )
        ]], colWidths=[W*0.28, W*0.72])
        att_box.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C["org_bg"]),
            ("BOX",           (0,0), (-1,-1), 1, C["orange"]),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("RIGHTPADDING",  (0,0), (-1,-1), 10),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(att_box)
    else:
        att_box = Table([[
            Paragraph("✓  ATTENDANCE OK", S("ao", fontName="Helvetica-Bold", fontSize=9, textColor=C["green"], leading=12)),
            Paragraph(
                f"Attendance {att_avg:.1f}% meets the 75% minimum. "
                f"Sem 1: {s1a}%  |  Sem 2: {s2a}%  |  Sem 3: {s3a}%",
                S("at2", fontName="Helvetica", fontSize=8.5, textColor=C["green"], leading=13)
            )
        ]], colWidths=[W*0.22, W*0.78])
        att_box.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C["grn_bg"]),
            ("BOX",           (0,0), (-1,-1), 1, C["grn_lt"]),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),("RIGHTPADDING", (0,0), (-1,-1), 10),
            ("TOPPADDING",    (0,0), (-1,-1), 8), ("BOTTOMPADDING",(0,0), (-1,-1), 8),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(att_box)
    story.append(Spacer(1, 18))

    # ── Subject scores table ─────────────────────────────────────────────────
    story.append(HR(sp_before=4, sp_after=6))
    story.append(Paragraph("Subject-wise Score Breakdown", sty["h1"]))
    story.append(Paragraph(
        "The table below shows your marks, cluster assignment, grade and pass/fail status for all 9 subjects across 3 semesters.",
        sty["body"]
    ))
    story.append(Spacer(1, 6))

    subj_hdr = [
        Paragraph("Subject", S("sh", fontName="Helvetica-Bold", fontSize=8.5, textColor=C["white"], leading=12)),
        Paragraph("Semester", S("sh", fontName="Helvetica-Bold", fontSize=8.5, textColor=C["white"], leading=12)),
        Paragraph("Cluster", S("sh", fontName="Helvetica-Bold", fontSize=8.5, textColor=C["white"], leading=12)),
        Paragraph("Score /100", S("sh", fontName="Helvetica-Bold", fontSize=8.5, textColor=C["white"], leading=12, alignment=TA_CENTER)),
        Paragraph("Grade", S("sh", fontName="Helvetica-Bold", fontSize=8.5, textColor=C["white"], leading=12, alignment=TA_CENTER)),
        Paragraph("Status", S("sh", fontName="Helvetica-Bold", fontSize=8.5, textColor=C["white"], leading=12, alignment=TA_CENTER)),
    ]
    subj_rows = [subj_hdr]
    subj_style = [
        ("BACKGROUND",    (0,0), (-1,0), C["navy"]),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("GRID",          (0,0), (-1,-1), 0.4, C["border"]),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [C["white"], C["bg1"]]),
        ("PADDING",       (0,0), (-1,-1), 7),
        ("ALIGN",         (3,0), (5,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]

    CLUSTER_BADGE_STYLE = {
        "Programming": sty["badge_b"],
        "Math/Logic":  sty["badge_g"],
        "Theory":      sty["badge_p"],
    }

    for i, (sname, col, clust, _, sem) in enumerate(SUBJECT_ROWS, start=1):
        sc  = int(student[col].values[0])
        g, gcolor, *_ = get_grade(sc)
        clr = colors.HexColor(gcolor.lstrip("#") and gcolor or "#64748b")
        try: clr = colors.HexColor(gcolor)
        except: clr = C["grey"]
        status     = "PASS" if sc >= 40 else "FAIL"
        status_clr = C["green"] if sc >= 40 else C["red"]
        badge_sty  = CLUSTER_BADGE_STYLE.get(clust, sty["body2"])
        subj_rows.append([
            Paragraph(sname, sty["body2"]),
            Paragraph(sem, sty["body"]),
            Paragraph(clust, badge_sty),
            Paragraph(str(sc), S(f"sc{i}", fontName="Helvetica-Bold", fontSize=10,
                                  textColor=clr, leading=13, alignment=TA_CENTER)),
            Paragraph(g, S(f"gr{i}", fontName="Helvetica-Bold", fontSize=9,
                            textColor=clr, leading=12, alignment=TA_CENTER)),
            Paragraph(status, S(f"st{i}", fontName="Helvetica-Bold", fontSize=8.5,
                                 textColor=status_clr, leading=12, alignment=TA_CENTER)),
        ])

    subj_table = Table(subj_rows,
                        colWidths=[W*0.26, W*0.10, W*0.17, W*0.14, W*0.10, W*0.13],
                        repeatRows=1)
    subj_table.setStyle(TableStyle(subj_style))
    story.append(subj_table)

    # ── Attendance detail table ───────────────────────────────────────────────
    story.append(HR(sp_before=18, sp_after=6))
    story.append(Paragraph("Semester-wise Attendance Detail", sty["h1"]))

    att_hdr_row = [
        Paragraph("", sty["body"]),
        Paragraph("Semester 1", S("ah", fontName="Helvetica-Bold", fontSize=9, textColor=C["white"], alignment=TA_CENTER, leading=12)),
        Paragraph("Semester 2", S("ah", fontName="Helvetica-Bold", fontSize=9, textColor=C["white"], alignment=TA_CENTER, leading=12)),
        Paragraph("Semester 3", S("ah", fontName="Helvetica-Bold", fontSize=9, textColor=C["white"], alignment=TA_CENTER, leading=12)),
        Paragraph("Average", S("ah", fontName="Helvetica-Bold", fontSize=9, textColor=C["white"], alignment=TA_CENTER, leading=12)),
        Paragraph("Min Required", S("ah", fontName="Helvetica-Bold", fontSize=9, textColor=C["white"], alignment=TA_CENTER, leading=12)),
    ]
    def att_val(v):
        clr = C["green"] if v >= 75 else (C["orange"] if v >= 65 else C["red"])
        return Paragraph(f"{v}%", S(f"av{v}", fontName="Helvetica-Bold", fontSize=11, textColor=clr,
                                     alignment=TA_CENTER, leading=14))
    att_tbl = Table(
        [att_hdr_row,
         ["Attendance", att_val(s1a), att_val(s2a), att_val(s3a),
          att_val(int(att_avg)), Paragraph("75%", S("mr", fontName="Helvetica-Bold", fontSize=11,
                                                      textColor=C["grey"], alignment=TA_CENTER, leading=14))]],
        colWidths=[W*0.18, W*0.145, W*0.145, W*0.145, W*0.145, W*0.14],
        rowHeights=[22, 36]
    )
    att_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), C["navy"]),
        ("BACKGROUND",    (0,1), (0,-1), C["bg2"]),
        ("FONTNAME",      (0,1), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",      (0,1), (0,-1), 9),
        ("TEXTCOLOR",     (0,1), (0,-1), C["text2"]),
        ("GRID",          (0,0), (-1,-1), 0.4, C["border"]),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("PADDING",       (0,0), (-1,-1), 7),
    ]))
    story.append(att_tbl)
    story.append(Spacer(1, 6))
    if att_avg < 75:
        story.append(Paragraph(
            f"⚠  Your attendance of {att_avg:.1f}% is below the 75% minimum. Immediate improvement is required "
            f"to avoid being barred from examinations in Semester 4.", sty["warn"]))
    else:
        story.append(Paragraph(
            f"✓  Your attendance of {att_avg:.1f}% satisfies the 75% minimum requirement. "
            f"Continue maintaining good attendance in Semester 4.", sty["ok"]))

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 2 — CLUSTER ANALYSIS + SEM 4 PREDICTION
    # ════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(banner)
    story.append(Spacer(1, 10))
    story.append(Paragraph("Cluster-wise Performance Analysis & Semester 4 Prediction", sty["title2"]))
    story.append(Paragraph(
        "Your subjects are grouped into 3 academic clusters. The Random Forest model analyses your "
        "performance trend across Semesters 1–3 and predicts your likely score in Semester 4 for each cluster.",
        sty["sub"]
    ))
    story.append(HR(thick=1, col=C["purple"], sp_before=0, sp_after=14))

    # ── Prediction summary table ─────────────────────────────────────────────
    story.append(Paragraph("Semester 4 Prediction Summary", sty["h1"]))
    ph = [Paragraph(t, S("ph", fontName="Helvetica-Bold", fontSize=8.5, textColor=C["white"], leading=12, alignment=TA_CENTER))
          for t in ["Cluster","Subjects Covered","Sem 1","Sem 2","Sem 3 Avg","Predicted Sem 4","Grade","Trend"]]
    pred_rows = [ph]
    pred_style = [
        ("BACKGROUND",    (0,0), (-1,0), C["purple"]),
        ("FONTSIZE",      (0,0), (-1,-1), 8.5),
        ("GRID",          (0,0), (-1,-1), 0.4, C["border"]),
        ("ROWBACKGROUNDS",(0,1), (-1,-2), [C["white"], C["bg1"]]),
        ("BACKGROUND",    (0,-1),(-1,-1), C["bg2"]),
        ("PADDING",       (0,0), (-1,-1), 6),
        ("ALIGN",         (2,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]
    for cluster, meta in CLUSTER_META.items():
        cur    = float(student[f"{cluster}_Avg"].values[0])
        pred   = float(predictions[cluster])
        s1_sc  = float(student[f"{cluster}_Sem1"].values[0])
        s2_sc  = float(student[f"{cluster}_Sem2"].values[0])
        g, gcolor_hex, *_ = get_grade(pred)
        try:    gc_col = colors.HexColor(gcolor_hex)
        except: gc_col = C["grey"]
        trend      = "↑ Improving" if pred > cur+1 else ("↓ Declining" if pred < cur-2 else "→ Stable")
        trend_col  = C["green"] if "↑" in trend else (C["red"] if "↓" in trend else C["grey"])
        subj_names = " / ".join(meta["subjects"].values())
        pred_rows.append([
            Paragraph(cluster.replace("_"," / "), sty["bold9"]),
            Paragraph(subj_names, S("sn", fontName="Helvetica", fontSize=7.5, textColor=C["grey"], leading=10)),
            Paragraph(f"{s1_sc:.0f}", S("sv", fontName="Helvetica", fontSize=9, textColor=C["text2"], alignment=TA_CENTER, leading=12)),
            Paragraph(f"{s2_sc:.0f}", S("sv2", fontName="Helvetica", fontSize=9, textColor=C["text2"], alignment=TA_CENTER, leading=12)),
            Paragraph(f"{cur:.1f}", S("sv3", fontName="Helvetica-Bold", fontSize=9, textColor=C["text"], alignment=TA_CENTER, leading=12)),
            Paragraph(f"{pred:.1f}", S("pv", fontName="Helvetica-Bold", fontSize=11, textColor=gc_col, alignment=TA_CENTER, leading=14)),
            Paragraph(g, S("gv", fontName="Helvetica-Bold", fontSize=9, textColor=gc_col, alignment=TA_CENTER, leading=12)),
            Paragraph(trend, S("tv", fontName="Helvetica-Bold", fontSize=8.5, textColor=trend_col, alignment=TA_CENTER, leading=12)),
        ])
    # Overall row
    og, ogc_hex, *_ = get_grade(pred_avg)
    try:    ogc = colors.HexColor(ogc_hex)
    except: ogc = C["grey"]
    o_trend = "↑ Improving" if pred_avg > sem3_avg+1 else ("↓ Declining" if pred_avg < sem3_avg-2 else "→ Stable")
    o_tc    = C["green"] if "↑" in o_trend else (C["red"] if "↓" in o_trend else C["grey"])
    pred_rows.append([
        Paragraph("OVERALL AVERAGE", S("ov", fontName="Helvetica-Bold", fontSize=8.5, textColor=C["text"], leading=12)),
        Paragraph("All 9 Subjects", S("as", fontName="Helvetica", fontSize=7.5, textColor=C["grey"], leading=10)),
        Paragraph("—", sty["body"]),
        Paragraph("—", sty["body"]),
        Paragraph(f"{sem3_avg:.1f}", S("os3", fontName="Helvetica-Bold", fontSize=9, textColor=C["text"], alignment=TA_CENTER, leading=12)),
        Paragraph(f"{pred_avg:.1f}", S("opv", fontName="Helvetica-Bold", fontSize=11, textColor=ogc, alignment=TA_CENTER, leading=14)),
        Paragraph(og, S("ogv", fontName="Helvetica-Bold", fontSize=9, textColor=ogc, alignment=TA_CENTER, leading=12)),
        Paragraph(o_trend, S("otv", fontName="Helvetica-Bold", fontSize=8.5, textColor=o_tc, alignment=TA_CENTER, leading=12)),
    ])

    pred_tbl = Table(pred_rows, colWidths=[W*0.16, W*0.22, W*0.07, W*0.07, W*0.10, W*0.13, W*0.09, W*0.13], repeatRows=1)
    pred_tbl.setStyle(TableStyle(pred_style))
    story.append(pred_tbl)
    story.append(Spacer(1, 20))

    # ── Per-cluster deep dive cards ──────────────────────────────────────────
    story.append(HR(sp_before=4, sp_after=6))
    story.append(Paragraph("Detailed Cluster Analysis", sty["h1"]))

    for cluster, meta in CLUSTER_META.items():
        c_main, c_lt, c_bg = CLUSTER_COLORS[cluster]
        cur    = float(student[f"{cluster}_Avg"].values[0])
        pred   = float(predictions[cluster])
        g, gc_hex, *_ = get_grade(cur)
        needs  = cur < 55 or pred < 55
        worst  = min(meta["cols"], key=lambda x: int(student[x[0]].values[0]))
        best   = max(meta["cols"], key=lambda x: int(student[x[0]].values[0]))
        w_sc   = int(student[worst[0]].values[0])
        b_sc   = int(student[best[0]].values[0])
        trend  = "Improving" if pred > cur+1 else ("Declining" if pred < cur-2 else "Stable")

        # cluster header row
        hdr = Table([[
            Paragraph(f"{meta['icon']}  {cluster.replace('_',' / ')} Cluster",
                      S("ch", fontName="Helvetica-Bold", fontSize=11, textColor=C["white"], leading=14)),
            Paragraph(f"Current Avg: {cur:.1f}  |  Predicted Sem 4: {pred:.1f}  |  Grade: {g}  |  Trend: {trend}",
                      S("ci", fontName="Helvetica", fontSize=8.5, textColor=colors.HexColor("#cbd5e1"), leading=12))
        ]], colWidths=[W*0.38, W*0.62])
        hdr.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C["navy"]),
            ("LEFTPADDING",   (0,0), (-1,-1), 12),
            ("RIGHTPADDING",  (0,0), (-1,-1), 12),
            ("TOPPADDING",    (0,0), (-1,-1), 9),
            ("BOTTOMPADDING", (0,0), (-1,-1), 9),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(hdr)

        # body
        body_rows = []

        # status + insight
        if needs:
            status_txt = (
                f"⚠  This cluster requires focused attention. Your current average of {cur:.1f} falls "
                f"below the 55-point improvement threshold. The Random Forest model predicts {pred:.1f} for "
                f"Semester 4 — this is based on your score trend and attendance pattern. "
                f"Your weakest subject is {worst[1]} ({w_sc}/100). "
                f"Prioritise this subject immediately to prevent a further decline."
            )
            status_sty = S("sn", fontName="Helvetica", fontSize=8.5, textColor=C["orange"], leading=13)
        else:
            status_txt = (
                f"✓  Performing well in this cluster. Your average of {cur:.1f} is above the target threshold. "
                f"The model predicts {pred:.1f} for Semester 4, reflecting a {'positive' if trend=='Improving' else 'stable'} trend. "
                f"Your strongest subject is {best[1]} ({b_sc}/100). "
                f"Continue this performance level and focus on revision during exam preparation."
            )
            status_sty = S("so", fontName="Helvetica", fontSize=8.5, textColor=C["green"], leading=13)

        body_rows.append([Paragraph(status_txt, status_sty)])

        # per-subject score rows
        subj_detail_hdr = [
            Paragraph("Subject", S("sdh", fontName="Helvetica-Bold", fontSize=8, textColor=C["grey"], leading=11)),
            Paragraph("Semester", S("sdh2", fontName="Helvetica-Bold", fontSize=8, textColor=C["grey"], leading=11)),
            Paragraph("Score / 100", S("sdh3", fontName="Helvetica-Bold", fontSize=8, textColor=C["grey"], alignment=TA_CENTER, leading=11)),
            Paragraph("Grade", S("sdh4", fontName="Helvetica-Bold", fontSize=8, textColor=C["grey"], alignment=TA_CENTER, leading=11)),
            Paragraph("Status", S("sdh5", fontName="Helvetica-Bold", fontSize=8, textColor=C["grey"], alignment=TA_CENTER, leading=11)),
        ]

        sem_map = {"C_Programming": "Sem 1", "Maths_I": "Sem 1", "Digital_Logic": "Sem 1",
                   "Data_Structures": "Sem 2", "Discrete_Maths": "Sem 2", "Computer_Organization": "Sem 2",
                   "Algorithms": "Sem 3", "Operating_Systems": "Sem 3", "DBMS": "Sem 3"}
        subj_det_rows = [subj_detail_hdr]
        for col, lbl in meta["cols"]:
            sc  = int(student[col].values[0])
            sg, sc_hex, *_ = get_grade(sc)
            try:    sc_col = colors.HexColor(sc_hex)
            except: sc_col = C["grey"]
            st_ = "Pass" if sc >= 40 else "Fail"
            st_c = C["green"] if sc >= 40 else C["red"]
            subj_det_rows.append([
                Paragraph(lbl, sty["body2"]),
                Paragraph(sem_map.get(col, ""), sty["body"]),
                Paragraph(str(sc), S(f"sc_{col}", fontName="Helvetica-Bold", fontSize=10,
                                      textColor=sc_col, alignment=TA_CENTER, leading=13)),
                Paragraph(sg, S(f"sg_{col}", fontName="Helvetica-Bold", fontSize=9,
                                 textColor=sc_col, alignment=TA_CENTER, leading=12)),
                Paragraph(st_, S(f"st_{col}", fontName="Helvetica-Bold", fontSize=8.5,
                                  textColor=st_c, alignment=TA_CENTER, leading=12)),
            ])

        det_tbl = Table(subj_det_rows,
                         colWidths=[W*0.35, W*0.15, W*0.18, W*0.14, W*0.12])
        det_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), c_bg),
            ("FONTSIZE",      (0,0), (-1,-1), 8.5),
            ("GRID",          (0,0), (-1,-1), 0.3, C["border"]),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [C["white"], c_bg]),
            ("PADDING",       (0,0), (-1,-1), 6),
            ("ALIGN",         (2,0), (-1,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]))

        card_content = Table(
            [[Paragraph(status_txt, status_sty)],
             [Spacer(1,6)],
             [det_tbl]],
            colWidths=[W]
        )
        card_content.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), c_bg),
            ("LEFTPADDING",   (0,0), (-1,-1), 12),
            ("RIGHTPADDING",  (0,0), (-1,-1), 12),
            ("TOPPADDING",    (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
            ("BOX",           (0,0), (-1,-1), 0.5, c_main),
        ]))
        story.append(KeepTogether([hdr, card_content]))
        story.append(Spacer(1, 10))

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 3 — RECOMMENDATIONS
    # ════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(banner)
    story.append(Spacer(1, 10))
    story.append(Paragraph("Personalised Learning Recommendations", sty["title2"]))

    all_recs = recommendations if recommendations else {
        c: {"youtube": YOUTUBE_RECOMMENDATIONS[c], "books": BOOK_RECOMMENDATIONS[c],
            "current_avg": float(student[f"{c}_Avg"].values[0]), "predicted": float(predictions[c])}
        for c in CLUSTER_META
    }
    rec_note = (
        "The following resources are recommended for clusters that need improvement (score < 55). "
        "YouTube videos and textbooks are curated specifically for B.Tech CSE students."
        if recommendations else
        "All clusters are performing above the threshold. The following enrichment resources are provided "
        "to help you excel further and prepare ahead for Semester 4."
    )
    story.append(Paragraph(rec_note, sty["sub"]))
    story.append(HR(thick=1, col=C["blue"], sp_before=0, sp_after=14))

    for cluster, rec in all_recs.items():
        meta = CLUSTER_META[cluster]
        c_main, c_lt, c_bg = CLUSTER_COLORS[cluster]
        cur  = rec.get("current_avg", float(student[f"{cluster}_Avg"].values[0]))
        pred = rec.get("predicted",   float(predictions[cluster]))

        # Section header
        rec_hdr = Table([[
            Paragraph(f"{meta['icon']}  {cluster.replace('_',' / ')} Cluster",
                      S("rch", fontName="Helvetica-Bold", fontSize=11, textColor=C["white"], leading=14)),
            Paragraph(f"Current Avg: {cur:.1f}  |  Predicted Sem 4: {pred:.1f}",
                      S("rci", fontName="Helvetica", fontSize=9, textColor=colors.HexColor("#cbd5e1"), leading=12,
                         alignment=TA_RIGHT))
        ]], colWidths=[W*0.5, W*0.5])
        rec_hdr.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C["navy"]),
            ("LEFTPADDING",   (0,0), (-1,-1), 12),("RIGHTPADDING", (0,0), (-1,-1), 12),
            ("TOPPADDING",    (0,0), (-1,-1), 9), ("BOTTOMPADDING",(0,0), (-1,-1), 9),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(rec_hdr)

        # YouTube + Books side by side
        yt_items  = []
        bk_items  = []
        for i, vid in enumerate(rec["youtube"], 1):
            yt_items.append(Paragraph(f"{i}. {vid['title']}",
                                       S(f"yt{i}", fontName="Helvetica-Bold", fontSize=8.5,
                                          textColor=C["text2"], leading=12)))
            yt_items.append(Paragraph(f"   {vid['url']}",
                                       S(f"ytl{i}", fontName="Helvetica", fontSize=7.5,
                                          textColor=C["blue_lt"], leading=11)))
            yt_items.append(Spacer(1, 3))

        for i, book in enumerate(rec["books"], 1):
            bk_items.append(Paragraph(f"{i}. {book['title']}",
                                       S(f"bk{i}", fontName="Helvetica-Bold", fontSize=8.5,
                                          textColor=C["text2"], leading=12)))
            bk_items.append(Paragraph(f"   Author: {book['author']}",
                                       S(f"bka{i}", fontName="Helvetica-Oblique", fontSize=8,
                                          textColor=C["grey"], leading=11)))
            bk_items.append(Spacer(1, 3))

        yt_hdr  = Paragraph("🎥  YouTube Resources", S("yth", fontName="Helvetica-Bold", fontSize=9,
                                                         textColor=C["blue"], leading=13))
        bk_hdr  = Paragraph("📖  Recommended Books", S("bkh", fontName="Helvetica-Bold", fontSize=9,
                                                         textColor=C["purple"], leading=13))

        col_l = [yt_hdr, Spacer(1,5)] + yt_items
        col_r = [bk_hdr, Spacer(1,5)] + bk_items

        rec_body = Table(
            [[col_l, col_r]],
            colWidths=[(W-6)*0.5, (W-6)*0.5]
        )
        rec_body.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), c_bg),
            ("BOX",           (0,0), (-1,-1), 0.5, c_main),
            ("LEFTPADDING",   (0,0), (-1,-1), 12),("RIGHTPADDING", (0,0), (-1,-1), 12),
            ("TOPPADDING",    (0,0), (-1,-1), 10),("BOTTOMPADDING",(0,0), (-1,-1), 10),
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
            ("LINEAFTER",     (0,0), (0,-1),  0.4, C["border"]),
        ]))
        story.append(rec_body)
        story.append(Spacer(1, 10))

    # ── Footer ───────────────────────────────────────────────────────────────
    story.append(HR(sp_before=16, sp_after=8, thick=0.5))
    story.append(Paragraph(
        f"Student Performance Prediction System  ·  B.Tech CSE  ·  Random Forest Model  ·  "
        f"Report generated on {datetime.now().strftime('%d %B %Y')}  ·  "
        f"Roll No: #{roll:02d}  ·  {name}",
        sty["foot"]
    ))

    doc.build(story)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "page"         not in st.session_state: st.session_state.page = "login"
if "student_data" not in st.session_state: st.session_state.student_data = None
if "login_error"  not in st.session_state: st.session_state.login_error = ""
if "is_admin"     not in st.session_state: st.session_state.is_admin = False
if "admin_view_roll" not in st.session_state: st.session_state.admin_view_roll = None


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 ── LOGIN
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.page == "login":

    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        st.markdown("""
        <div style='padding-top:50px; text-align:center;'>
            <div style='font-size:3.5rem; margin-bottom:10px;'>🎓</div>
            <h1 style='font-size:1.7rem; font-weight:800; background:linear-gradient(135deg,#60a5fa,#a78bfa,#34d399);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0 0 8px;'>
                Student Performance<br>Prediction System
            </h1>
            <p style='color:#475569; font-size:0.85rem; margin-bottom:28px;'>B.Tech CSE · Random Forest · 3 Cluster Analysis</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Login type tabs ───────────────────────────────────────────────────
        tab_student, tab_admin = st.tabs(["🎓  Student Login", "🛡️  Admin Login"])

        # ── STUDENT TAB ───────────────────────────────────────────────────────
        with tab_student:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            roll_no  = st.number_input("Roll Number", min_value=1, max_value=20, value=1, step=1, key="s_roll")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="s_pass")

            col_btn, _ = st.columns([1, 1])
            with col_btn:
                login_btn = st.button("🚀  Login & View Report", use_container_width=True, key="s_login")

            if st.session_state.login_error and not st.session_state.is_admin:
                st.markdown(f"<div class='error-msg'>❌ {st.session_state.login_error}</div>", unsafe_allow_html=True)

            if login_btn:
                s = merged[merged["Roll_No"] == roll_no]
                if s.empty:
                    st.session_state.login_error = "Roll number not found."
                elif not password.strip():
                    st.session_state.login_error = "Please enter your password."
                elif password.strip() != STUDENT_PASSWORDS.get(roll_no, ""):
                    st.session_state.login_error = f"Incorrect password for Roll No {roll_no}."
                else:
                    st.session_state.login_error  = ""
                    st.session_state.is_admin     = False
                    st.session_state.student_data = s
                    st.session_state.page         = "dashboard"
                    st.rerun()

        # ── ADMIN TAB ─────────────────────────────────────────────────────────
        with tab_admin:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            admin_user = st.text_input("Admin Username", placeholder="Enter username", key="a_user")
            admin_pass = st.text_input("Admin Password", type="password", placeholder="Enter password", key="a_pass")

            col_btn2, _ = st.columns([1, 1])
            with col_btn2:
                admin_btn = st.button("🛡️  Admin Login", use_container_width=True, key="a_login")

            if st.session_state.login_error and st.session_state.is_admin:
                st.markdown(f"<div class='error-msg'>❌ {st.session_state.login_error}</div>", unsafe_allow_html=True)

            # Admin credentials — change these as needed
            ADMIN_USERNAME = "admin"
            ADMIN_PASSWORD = "admin123"

            if admin_btn:
                if not admin_user.strip() or not admin_pass.strip():
                    st.session_state.login_error = "Please enter username and password."
                    st.session_state.is_admin = True
                elif admin_user.strip() != ADMIN_USERNAME or admin_pass.strip() != ADMIN_PASSWORD:
                    st.session_state.login_error = "Incorrect admin credentials."
                    st.session_state.is_admin = True
                else:
                    st.session_state.login_error     = ""
                    st.session_state.is_admin        = True
                    st.session_state.admin_view_roll = None
                    st.session_state.page            = "admin"
                    st.rerun()



        st.markdown("""
        <div style='text-align:center; color:#334155; font-size:0.75rem; margin-top:24px; padding-bottom:40px;'>
            20 Students · 3 Semesters · 9 Subjects · AI Prediction
        </div>""", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 ── DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "dashboard":

    student  = st.session_state.student_data
    roll     = int(student["Roll_No"].values[0])
    name     = student["Name"].values[0]
    preds    = predict_student(roll, merged, models)
    recs     = get_recommendations(student, preds)
    sem3_avg = float(student["Sem3_Avg"].values[0])
    att_avg  = float(student["Attendance_Avg"].values[0])
    pred_avg = float(np.mean(list(preds.values())))
    grade, gcolor, gbg, gborder = get_grade(sem3_avg)
    pred_grade, pgcolor, *_ = get_grade(pred_avg)
    a_stat, a_color = att_status(att_avg)

    # Nav bar
    c1, c2 = st.columns([5, 1])
    with c1:
        st.markdown(f"""
        <div style='background:linear-gradient(90deg,#0d1526,#091020); border:1px solid #1e3a5f;
                    border-radius:12px; padding:13px 24px; display:flex; align-items:center; gap:12px; margin-bottom:24px;'>
            <span style='color:#60a5fa; font-weight:700; font-size:1rem;'>🎓 SPP System</span>
            <span style='color:#1e3a5f;'>|</span>
            <span style='color:#94a3b8; font-size:0.85rem;'><strong style='color:#e2e8f0;'>{name}</strong> &nbsp;·&nbsp; Roll No: {roll:02d} &nbsp;·&nbsp; B.Tech CSE</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        if st.button("← Logout"):
            st.session_state.page = "login"; st.session_state.student_data = None; st.rerun()

    # Metric cards
    st.markdown(f"""
    <div class='metric-row'>
        <div class='metric-card mc-blue'>
            <span class='metric-icon'>🎯</span>
            <div class='metric-value mv-blue'>{grade}</div>
            <div class='metric-label'>Current Grade</div>
        </div>
        <div class='metric-card mc-purple'>
            <span class='metric-icon'>🔮</span>
            <div class='metric-value mv-purple'>{pred_avg:.1f}</div>
            <div class='metric-label'>Predicted Sem 4 Avg</div>
        </div>
        <div class='metric-card mc-green'>
            <span class='metric-icon'>📅</span>
            <div class='metric-value mv-green'>{att_avg:.0f}%</div>
            <div class='metric-label'>Attendance · {a_stat}</div>
        </div>
        <div class='metric-card mc-orange'>
            <span class='metric-icon'>📊</span>
            <div class='metric-value mv-orange'>{sem3_avg:.1f}</div>
            <div class='metric-label'>Sem 3 Avg Score</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Attendance alert
    if att_avg < 75:
        st.markdown(
            f'<div style="background:rgba(251,191,36,0.07); border:1px solid rgba(251,191,36,0.25); border-radius:10px; padding:12px 18px; color:#fbbf24; font-size:0.85rem; margin:10px 0;">'
            f'⚠️ <strong>Attendance Alert:</strong> {att_avg:.1f}% is below the 75% minimum. This may affect exam eligibility.'
            f'</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div style="background:rgba(52,211,153,0.07); border:1px solid rgba(52,211,153,0.25); border-radius:10px; padding:12px 18px; color:#34d399; font-size:0.85rem; margin:10px 0;">'
            f'✅ <strong>Attendance Good:</strong> {att_avg:.1f}% — meets the 75% requirement.'
            f'</div>', unsafe_allow_html=True)

    # Subject table
    st.markdown("<p class='section-title'>Subject Marks, Grades & Status</p>", unsafe_allow_html=True)
    rows_html = ""
    for sname, col, clust, badge_cls, sem in SUBJECT_ROWS:
        sc = int(student[col].values[0])
        g, gc, gbg2, gbd = get_grade(sc)
        status_html = (
            '<span style="color:#34d399; font-size:0.78rem; font-weight:600;">✓ Pass</span>'
            if sc >= 40 else
            '<span style="color:#f87171; font-size:0.78rem; font-weight:600;">✗ Fail</span>'
        )
        rows_html += (
            f'<div class="subject-row">'
            f'<div class="subj-name">{sname}</div>'
            f'<div><span class="subj-cluster-badge {badge_cls}">{clust}</span></div>'
            f'<div style="color:#475569; font-size:0.8rem;">{sem}</div>'
            f'<div class="subj-score">{sc}</div>'
            f'<div style="display:flex; align-items:center; gap:8px;">'
            f'<span class="grade-badge" style="background:{gbg2}; color:{gc}; border:1px solid {gbd};">{g}</span>'
            f'{status_html}'
            f'</div>'
            f'</div>'
        )

    st.markdown(
        f'<div class="subject-table">'
        f'<div class="subject-table-header">'
        f'<div>Subject</div><div>Cluster</div><div>Semester</div><div>Score</div><div>Grade / Status</div>'
        f'</div>'
        f'{rows_html}'
        f'</div>',
        unsafe_allow_html=True
    )

    # Charts
    st.markdown("<p class='section-title'>Visual Analysis</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(chart_attendance(student), use_container_width=True)
    with c2: st.plotly_chart(chart_cluster_pie(student), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3: st.plotly_chart(chart_performance_trend(student), use_container_width=True)
    with c4: st.plotly_chart(chart_predicted(student, preds), use_container_width=True)

    # Go to report
    st.markdown("<p class='section-title'>Detailed Report</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#0a1220; border:1px solid #1e3a5f; border-radius:12px; padding:20px 24px;
                display:flex; align-items:center; justify-content:space-between; margin-bottom:16px;'>
        <div>
            <div style='color:#e2e8f0; font-weight:600; margin-bottom:4px;'>📋 Full Cluster Analysis & Recommendations</div>
            <div style='color:#475569; font-size:0.8rem;'>Detailed insights · YouTube resources · Book suggestions · PDF download</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("📋  View Detailed Report & Recommendations →"):
        st.session_state.page = "report"; st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 ── DETAILED REPORT
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "report":

    student  = st.session_state.student_data
    roll     = int(student["Roll_No"].values[0])
    name     = student["Name"].values[0]
    preds    = predict_student(roll, merged, models)
    recs     = get_recommendations(student, preds)
    sem3_avg = float(student["Sem3_Avg"].values[0])
    att_avg  = float(student["Attendance_Avg"].values[0])

    # Nav
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.markdown(f"""
        <div style='background:linear-gradient(90deg,#0d1526,#091020); border:1px solid #1e3a5f;
                    border-radius:12px; padding:13px 24px; margin-bottom:24px;'>
            <span style='color:#60a5fa; font-weight:700;'>🎓 SPP System</span>
            <span style='color:#1e3a5f; margin:0 10px;'>|</span>
            <span style='color:#94a3b8; font-size:0.85rem;'><strong style='color:#e2e8f0;'>{name}</strong> · Detailed Report</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        if st.button("← Dashboard"): st.session_state.page = "dashboard"; st.rerun()
    with c3:
        if st.button("⬅ Logout"):    st.session_state.page = "login"; st.session_state.student_data = None; st.rerun()

    st.markdown(f"""
    <h2 style='color:#e2e8f0; font-size:1.4rem; font-weight:700; margin-bottom:4px;'>📋 Detailed Report — {name}</h2>
    <p style='color:#475569; font-size:0.82rem; margin-bottom:24px;'>Cluster-wise analysis · Predicted Semester 4 · Personalized learning recommendations</p>
    """, unsafe_allow_html=True)

    # Cluster analysis cards
    st.markdown("<p class='section-title'>Cluster-wise Performance Analysis</p>", unsafe_allow_html=True)

    # Inline style maps — Streamlit strips class= attributes, so we use styles directly
    CC_STYLES = {
        "cc-prog":   "background:linear-gradient(135deg,#0d1f3c,#0a1828); border:1px solid rgba(59,130,246,0.25); border-radius:14px; padding:24px 28px; margin-bottom:16px;",
        "cc-theory": "background:linear-gradient(135deg,#150d2e,#100a22); border:1px solid rgba(139,92,246,0.25); border-radius:14px; padding:24px 28px; margin-bottom:16px;",
        "cc-math":   "background:linear-gradient(135deg,#082819,#061d12); border:1px solid rgba(52,211,153,0.25); border-radius:14px; padding:24px 28px; margin-bottom:16px;",
    }
    SCORE_STYLES = {
        "cc-score-prog":   "font-size:1.8rem; font-weight:800; font-family:JetBrains Mono,monospace; color:#60a5fa;",
        "cc-score-theory": "font-size:1.8rem; font-weight:800; font-family:JetBrains Mono,monospace; color:#a78bfa;",
        "cc-score-math":   "font-size:1.8rem; font-weight:800; font-family:JetBrains Mono,monospace; color:#34d399;",
    }
    PB_STYLES = {
        "pb-prog":   "height:100%; border-radius:999px; background:linear-gradient(90deg,#3b82f6,#60a5fa);",
        "pb-theory": "height:100%; border-radius:999px; background:linear-gradient(90deg,#7c3aed,#a78bfa);",
        "pb-math":   "height:100%; border-radius:999px; background:linear-gradient(90deg,#059669,#34d399);",
    }

    for cluster, meta in CLUSTER_META.items():
        cur   = float(student[f"{cluster}_Avg"].values[0])
        pred  = float(preds[cluster])
        g, gc, *_ = get_grade(cur)
        needs = cur < 55 or pred < 55
        pct   = min(100, int(cur))

        worst = min(meta["cols"], key=lambda x: int(student[x[0]].values[0]))
        best  = max(meta["cols"], key=lambda x: int(student[x[0]].values[0]))

        insight = (
            f"This cluster requires focused attention. Average of {cur:.1f} suggests gaps in core concepts. "
            f"Weakest subject: <b>{worst[1]}</b> ({int(student[worst[0]].values[0])}). "
            f"Predicted Sem 4 score: {pred:.1f}. Improving here will significantly boost overall CGPA."
        ) if needs else (
            f"Performing well with an average of {cur:.1f}. Strongest: <b>{best[1]}</b> ({int(student[best[0]].values[0])}). "
            f"Predicted Sem 4 score: {pred:.1f}. Maintain consistent practice across all topics."
        )

        if needs:
            alert_html = (
                f'<div style="background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.2); '
                f'border-radius:8px; padding:9px 14px; color:#fbbf24; font-size:0.78rem; margin-bottom:12px;">'
                f'⚠️ Needs Attention — target {max(cur+10, 60):.0f}+ in Semester 4</div>'
            )
        else:
            alert_html = (
                f'<div style="background:rgba(52,211,153,0.08); border:1px solid rgba(52,211,153,0.2); '
                f'border-radius:8px; padding:9px 14px; color:#34d399; font-size:0.78rem; margin-bottom:12px;">'
                f'✅ On Track — predicted {pred:.1f} for Semester 4</div>'
            )

        mini_bars = ""
        for col, lbl in meta["cols"]:
            sc = int(student[col].values[0])
            _, sgc, *_ = get_grade(sc)
            sg, *_ = get_grade(sc)
            bar_color = meta["color"]
            mini_bars += (
                f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:7px;">'
                f'<div style="color:#64748b; font-size:0.78rem; width:130px;">{lbl}</div>'
                f'<div style="flex:1; background:rgba(255,255,255,0.05); border-radius:999px; height:5px; overflow:hidden;">'
                f'<div style="width:{sc}%; height:100%; background:{bar_color}; border-radius:999px; opacity:0.8;"></div>'
                f'</div>'
                f'<div style="font-family:JetBrains Mono,monospace; font-size:0.78rem; color:{sgc}; width:28px; text-align:right;">{sc}</div>'
                f'<div style="font-family:JetBrains Mono,monospace; font-size:0.72rem; color:{sgc}; width:20px;">{sg}</div>'
                f'</div>'
            )

        cc_cls      = meta["cc"]
        score_cls   = meta["score_cls"]
        pb_cls      = meta["pb"]
        cl_color    = meta["color"]
        cl_icon     = meta["icon"]
        cl_name     = cluster.replace("_", " / ")
        cl_subjects = " · ".join(meta["subjects"].values())

        # Build card with fully inline styles (no class= attributes — Streamlit strips them)
        card_bg    = CC_STYLES[cc_cls]
        score_style = SCORE_STYLES[score_cls]
        pb_gradient = PB_STYLES[pb_cls].replace("{pct}", str(pct))

        card_html = (
            f'<div style="{card_bg}">'
            f'<div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:14px;">'
            f'<div>'
            f'<div style="font-size:1.5rem; margin-bottom:4px;">{cl_icon}</div>'
            f'<div style="font-size:1rem; font-weight:700; color:#e2e8f0;">{cl_name} Cluster</div>'
            f'<div style="color:{cl_color}; font-size:0.76rem; margin-top:2px;">{cl_subjects}</div>'
            f'</div>'
            f'<div style="text-align:right;">'
            f'<div style="{score_style}">{cur:.1f}</div>'
            f'<div style="color:#475569; font-size:0.7rem;">Current Avg</div>'
            f'<div style="color:{cl_color}; font-size:0.85rem; font-weight:700; margin-top:6px;">Sem 4: {pred:.1f}</div>'
            f'</div>'
            f'</div>'
            f'<div style="background:rgba(255,255,255,0.06); border-radius:999px; height:5px; margin:8px 0 14px; overflow:hidden;">'
            f'<div style="{pb_gradient}"></div>'
            f'</div>'
            f'{alert_html}'
            f'<div style="color:#94a3b8; font-size:0.82rem; line-height:1.65; margin-bottom:12px;">{insight}</div>'
            f'<div style="border-top:1px solid rgba(255,255,255,0.06); padding-top:12px;">'
            f'<div style="color:#475569; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">Subject Breakdown</div>'
            f'{mini_bars}'
            f'</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

    # Recommendations
    st.markdown("<p class='section-title'>📌 YouTube & Book Recommendations</p>", unsafe_allow_html=True)

    clusters_to_show = recs if recs else {c: {"youtube": YOUTUBE_RECOMMENDATIONS[c],
                                               "books": BOOK_RECOMMENDATIONS[c],
                                               "current_avg": float(student[f"{c}_Avg"].values[0]),
                                               "predicted": float(preds[c])} for c in CLUSTER_META}
    if not recs:
        st.markdown("<div class='alert-ok'>🌟 <strong>All clusters performing well!</strong> Showing enrichment resources below.</div>", unsafe_allow_html=True)

    for cluster, rec_data in clusters_to_show.items():
        meta = CLUSTER_META[cluster]
        with st.expander(f"{meta['icon']} {cluster.replace('_',' / ')} — Current: {rec_data['current_avg']:.1f}  ·  Predicted Sem 4: {rec_data['predicted']:.1f}", expanded=True):
            cy, cb = st.columns(2)
            with cy:
                st.markdown("**🎥 YouTube Resources**")
                for vid in rec_data["youtube"]:
                    st.markdown(f"<div class='rec-item'><div class='rec-item-title'>▶ {vid['title']}</div><a class='rec-item-link' href='{vid['url']}' target='_blank'>{vid['url']}</a></div>", unsafe_allow_html=True)
            with cb:
                st.markdown("**📖 Recommended Books**")
                for book in rec_data["books"]:
                    st.markdown(f"<div class='rec-item'><div class='rec-item-title'>📕 {book['title']}</div><div class='rec-item-meta'>by {book['author']}</div></div>", unsafe_allow_html=True)

    # PDF Download
    st.markdown("<p class='section-title'>⬇️ Download Report as PDF</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#0a1220; border:1px dashed #1e3a5f; border-radius:12px; padding:22px 28px; text-align:center; margin-bottom:16px;'>
        <div style='font-size:2rem; margin-bottom:8px;'>📄</div>
        <div style='color:#94a3b8; font-size:0.9rem; margin-bottom:4px;'>Download Complete Student Report as PDF</div>
        <div style='color:#475569; font-size:0.78rem;'>Student info · Subject scores · Attendance · Cluster analysis · Predictions · Recommendations</div>
    </div>
    """, unsafe_allow_html=True)

    pdf_buf = generate_pdf(student, preds, recs)
    st.download_button(
        label="📥  Download PDF Report",
        data=pdf_buf,
        file_name=f"SPP_Report_{name.replace(' ','_')}_Roll{roll:02d}.pdf",
        mime="application/pdf"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 ── ADMIN PANEL
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "admin":

    # ── Nav bar ───────────────────────────────────────────────────────────────
    c1, c2 = st.columns([5, 1])
    with c1:
        st.markdown("""
        <div style='background:linear-gradient(90deg,#1a0533,#0d1526); border:1px solid #3b1f6b;
                    border-radius:12px; padding:13px 24px; display:flex; align-items:center;
                    gap:12px; margin-bottom:24px;'>
            <span style='color:#a78bfa; font-weight:700; font-size:1rem;'>🛡️ Admin Panel</span>
            <span style='color:#2d1b5e;'>|</span>
            <span style='color:#94a3b8; font-size:0.85rem;'>
                <strong style='color:#e2e8f0;'>Administrator</strong>
                &nbsp;·&nbsp; Full Student Access &nbsp;·&nbsp; B.Tech CSE
            </span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        if st.button("← Logout", key="admin_logout"):
            st.session_state.page         = "login"
            st.session_state.is_admin     = False
            st.session_state.admin_view_roll = None
            st.session_state.student_data = None
            st.rerun()

    # ── Heading ───────────────────────────────────────────────────────────────
    st.markdown("""
    <h2 style='color:#e2e8f0; font-size:1.4rem; font-weight:700; margin-bottom:4px;'>
        📋 All Students Overview
    </h2>
    <p style='color:#475569; font-size:0.82rem; margin-bottom:20px;'>
        Click any student row to view their full report
    </p>
    """, unsafe_allow_html=True)

    # ── Summary metric cards ──────────────────────────────────────────────────
    total_students  = len(merged)
    avg_overall     = float(merged["Sem3_Avg"].mean())
    avg_attendance  = float(merged["Attendance_Avg"].mean())
    at_risk         = int((merged["Sem3_Avg"] < 50).sum())

    st.markdown(f"""
    <div class='metric-row'>
        <div class='metric-card mc-blue'>
            <span class='metric-icon'>👥</span>
            <div class='metric-value mv-blue'>{total_students}</div>
            <div class='metric-label'>Total Students</div>
        </div>
        <div class='metric-card mc-green'>
            <span class='metric-icon'>📊</span>
            <div class='metric-value mv-green'>{avg_overall:.1f}</div>
            <div class='metric-label'>Class Avg (Sem 3)</div>
        </div>
        <div class='metric-card mc-purple'>
            <span class='metric-icon'>📅</span>
            <div class='metric-value mv-purple'>{avg_attendance:.0f}%</div>
            <div class='metric-label'>Avg Attendance</div>
        </div>
        <div class='metric-card mc-orange'>
            <span class='metric-icon'>⚠️</span>
            <div class='metric-value mv-orange'>{at_risk}</div>
            <div class='metric-label'>At Risk (&lt;50 avg)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Class-wide Charts ─────────────────────────────────────────────────────
    st.markdown("<p class='section-title'>📊 Class Performance Overview</p>", unsafe_allow_html=True)

    # Chart 1 + 2: Bar chart of all students + Cluster averages
    c1, c2 = st.columns(2)

    with c1:
        # All students Sem3 avg bar chart
        names_list  = [n.strip().split()[0] for n in merged["Name"].tolist()]  # first name only
        s3_avgs     = merged["Sem3_Avg"].tolist()
        bar_colors  = ["#34d399" if v >= 60 else ("#fbbf24" if v >= 50 else "#f87171") for v in s3_avgs]
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=names_list, y=s3_avgs,
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{v:.0f}" for v in s3_avgs],
            textposition="outside",
            textfont=dict(size=9, color="#94a3b8"),
        ))
        fig_bar.add_hline(y=50, line_dash="dot", line_color="#fbbf24", line_width=1.5,
                          annotation_text="  Pass (50)", annotation_font=dict(color="#fbbf24", size=10))
        fig_bar.update_layout(
            title=dict(text="Sem 3 Average — All Students", font=dict(size=13, color="#94a3b8")),
            xaxis=dict(tickangle=-45, tickfont=dict(size=8), gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(range=[0, 110], gridcolor="#0d1829"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            margin=dict(l=16, r=16, t=44, b=60),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        # Cluster averages grouped bar
        prog_avgs  = merged["Programming_Avg"].tolist()
        theory_avgs= merged["Theory_Avg"].tolist()
        math_avgs  = merged["Math_Logic_Avg"].tolist()
        fig_clust = go.Figure()
        fig_clust.add_trace(go.Bar(name="Programming", x=names_list, y=prog_avgs,
                                    marker_color="#3b82f6", opacity=0.85))
        fig_clust.add_trace(go.Bar(name="Theory",      x=names_list, y=theory_avgs,
                                    marker_color="#7c3aed", opacity=0.85))
        fig_clust.add_trace(go.Bar(name="Math/Logic",  x=names_list, y=math_avgs,
                                    marker_color="#059669", opacity=0.85))
        fig_clust.update_layout(
            title=dict(text="Cluster Averages — All Students", font=dict(size=13, color="#94a3b8")),
            barmode="group",
            xaxis=dict(tickangle=-45, tickfont=dict(size=8), gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(range=[0, 110], gridcolor="#0d1829"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            margin=dict(l=16, r=16, t=44, b=60),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e3a5f", borderwidth=1,
                        font=dict(size=10, color="#94a3b8")),
        )
        st.plotly_chart(fig_clust, use_container_width=True)

    # Chart 3 + 4: Attendance + Sem trend line
    c3, c4 = st.columns(2)

    with c3:
        # Attendance bar for all students
        att_vals   = merged["Attendance_Avg"].tolist()
        att_colors = ["#34d399" if v >= 75 else ("#fbbf24" if v >= 65 else "#f87171") for v in att_vals]
        fig_att = go.Figure()
        fig_att.add_trace(go.Bar(
            x=names_list, y=att_vals,
            marker=dict(color=att_colors, line=dict(width=0)),
            text=[f"{v:.0f}%" for v in att_vals],
            textposition="outside",
            textfont=dict(size=9, color="#94a3b8"),
        ))
        fig_att.add_hline(y=75, line_dash="dot", line_color="#fbbf24", line_width=1.5,
                          annotation_text="  75% Min", annotation_font=dict(color="#fbbf24", size=10))
        fig_att.update_layout(
            title=dict(text="Attendance % — All Students", font=dict(size=13, color="#94a3b8")),
            xaxis=dict(tickangle=-45, tickfont=dict(size=8), gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(range=[0, 115], gridcolor="#0d1829"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            margin=dict(l=16, r=16, t=44, b=60),
            showlegend=False,
        )
        st.plotly_chart(fig_att, use_container_width=True)

    with c4:
        # Class average trend Sem1 → Sem2 → Sem3
        class_s1 = float(merged["Sem1_Avg"].mean())
        class_s2 = float(merged["Sem2_Avg"].mean())
        class_s3 = float(merged["Sem3_Avg"].mean())
        # per-student trend lines (faint)
        fig_trend = go.Figure()
        for _, row in merged.iterrows():
            fig_trend.add_trace(go.Scatter(
                x=["Sem 1", "Sem 2", "Sem 3"],
                y=[float(row["Sem1_Avg"]), float(row["Sem2_Avg"]), float(row["Sem3_Avg"])],
                mode="lines",
                line=dict(color="#1e3a5f", width=1),
                showlegend=False, hoverinfo="skip",
            ))
        # class average line on top
        fig_trend.add_trace(go.Scatter(
            x=["Sem 1", "Sem 2", "Sem 3"],
            y=[class_s1, class_s2, class_s3],
            mode="lines+markers+text",
            line=dict(color="#60a5fa", width=3),
            marker=dict(size=10, color="#60a5fa", line=dict(color="#060b14", width=2)),
            text=[f"{class_s1:.1f}", f"{class_s2:.1f}", f"{class_s3:.1f}"],
            textposition="top center",
            textfont=dict(size=11, color="#60a5fa"),
            name="Class Avg",
        ))
        fig_trend.update_layout(
            title=dict(text="Semester Trend — Class Average vs Each Student", font=dict(size=13, color="#94a3b8")),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(range=[0, 110], gridcolor="#0d1829"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            margin=dict(l=16, r=16, t=44, b=16),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e3a5f", borderwidth=1,
                        font=dict(size=10, color="#94a3b8")),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # ── Semester filter ───────────────────────────────────────────────────────
    st.markdown("<p class='section-title'>Student Results Table</p>", unsafe_allow_html=True)

    sem_filter = st.selectbox(
        "View Semester",
        ["All Semesters", "Semester 1", "Semester 2", "Semester 3"],
        key="admin_sem_filter"
    )

    # ── Build table ───────────────────────────────────────────────────────────
    SEM_COLS = {
        "Semester 1": ["C_Programming", "Maths_I", "Digital_Logic", "Sem1_Attendance"],
        "Semester 2": ["Data_Structures", "Discrete_Maths", "Computer_Organization", "Sem2_Attendance"],
        "Semester 3": ["Algorithms", "Operating_Systems", "DBMS", "Sem3_Attendance"],
    }
    SEM_LABELS = {
        "C_Programming": "C Prog", "Maths_I": "Maths I", "Digital_Logic": "Dig. Logic",
        "Sem1_Attendance": "Attend %",
        "Data_Structures": "Data Struct", "Discrete_Maths": "Disc. Maths",
        "Computer_Organization": "Comp. Org", "Sem2_Attendance": "Attend %",
        "Algorithms": "Algorithms", "Operating_Systems": "OS",
        "DBMS": "DBMS", "Sem3_Attendance": "Attend %",
    }

    # Header
    if sem_filter == "All Semesters":
        header_html = (
            "<div style='display:grid; grid-template-columns:40px 180px 80px 80px 80px 80px 80px; "
            "gap:6px; padding:10px 20px; background:#0a0a0f; border-bottom:1px solid #1e3a5f; "
            "font-size:.65rem; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:1px;'>"
            "<div>#</div><div>Student Name</div>"
            "<div style='text-align:center'>Sem 1 Avg</div>"
            "<div style='text-align:center'>Sem 2 Avg</div>"
            "<div style='text-align:center'>Sem 3 Avg</div>"
            "<div style='text-align:center'>Attendance</div>"
            "<div style='text-align:center'>Action</div>"
            "</div>"
        )
    else:
        sub_cols = SEM_COLS[sem_filter]
        sub_labels = [SEM_LABELS[c] for c in sub_cols]
        col_widths = "40px 180px " + " ".join(["70px"] * len(sub_cols)) + " 80px"
        header_html = (
            f"<div style='display:grid; grid-template-columns:{col_widths}; "
            f"gap:6px; padding:10px 20px; background:#0a0a0f; border-bottom:1px solid #1e3a5f; "
            f"font-size:.65rem; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:1px;'>"
            f"<div>#</div><div>Student Name</div>"
            + "".join(f"<div style='text-align:center'>{l}</div>" for l in sub_labels)
            + "<div style='text-align:center'>Action</div>"
            "</div>"
        )

    st.markdown(
        f"<div style='background:#0d111a; border:1px solid #1e3a5f; border-radius:14px; overflow:hidden; margin-bottom:24px;'>"
        f"{header_html}",
        unsafe_allow_html=True
    )

    for _, row in merged.iterrows():
        roll   = int(row["Roll_No"])
        name   = row["Name"].strip()
        s1_avg = float(row["Sem1_Avg"])
        s2_avg = float(row["Sem2_Avg"])
        s3_avg = float(row["Sem3_Avg"])
        att    = float(row["Attendance_Avg"])

        def score_badge(sc):
            g, gc, gbg2, gbd = get_grade(sc)
            return (
                f'<span style="background:{gbg2}; color:{gc}; border:1px solid {gbd}; '
                f'padding:2px 8px; border-radius:5px; font-size:.75rem; '
                f'font-weight:700; font-family:monospace;">{sc:.0f} {g}</span>'
            )

        att_col = "#34d399" if att >= 75 else ("#fbbf24" if att >= 65 else "#f87171")

        if sem_filter == "All Semesters":
            row_html = (
                f"<div style='display:grid; grid-template-columns:40px 180px 80px 80px 80px 80px 80px; "
                f"gap:6px; padding:11px 20px; border-bottom:1px solid #0d1829; align-items:center;'>"
                f"<div style='color:#334155; font-size:.75rem; font-weight:600;'>{roll:02d}</div>"
                f"<div style='color:#e2e8f0; font-size:.82rem; font-weight:500;'>{name}</div>"
                f"<div style='text-align:center;'>{score_badge(s1_avg)}</div>"
                f"<div style='text-align:center;'>{score_badge(s2_avg)}</div>"
                f"<div style='text-align:center;'>{score_badge(s3_avg)}</div>"
                f"<div style='text-align:center; color:{att_col}; font-size:.8rem; font-weight:700;'>{att:.0f}%</div>"
                f"<div style='text-align:center;'>__BTN_{roll}__</div>"
                f"</div>"
            )
        else:
            sub_cols = SEM_COLS[sem_filter]
            col_widths = "40px 180px " + " ".join(["70px"] * len(sub_cols)) + " 80px"
            subject_cells = "".join(
                f"<div style='text-align:center;'>{score_badge(float(row[c]))}</div>"
                for c in sub_cols[:-1]
            )
            att_val = float(row[sub_cols[-1]])
            att_c   = "#34d399" if att_val >= 75 else ("#fbbf24" if att_val >= 65 else "#f87171")
            row_html = (
                f"<div style='display:grid; grid-template-columns:{col_widths}; "
                f"gap:6px; padding:11px 20px; border-bottom:1px solid #0d1829; align-items:center;'>"
                f"<div style='color:#334155; font-size:.75rem; font-weight:600;'>{roll:02d}</div>"
                f"<div style='color:#e2e8f0; font-size:.82rem; font-weight:500;'>{name}</div>"
                + subject_cells
                + f"<div style='text-align:center; color:{att_c}; font-size:.8rem; font-weight:700;'>{att_val:.0f}%</div>"
                f"<div style='text-align:center;'>__BTN_{roll}__</div>"
                f"</div>"
            )

        # Render row HTML (placeholder for button)
        btn_placeholder = f"__BTN_{roll}__"
        before_btn = row_html[:row_html.find(btn_placeholder)]
        after_btn  = row_html[row_html.find(btn_placeholder) + len(btn_placeholder):]

        col_row, col_btn = st.columns([6, 1])
        with col_row:
            st.markdown(before_btn + after_btn, unsafe_allow_html=True)
        with col_btn:
            if st.button("View →", key=f"view_{roll}"):
                st.session_state.admin_view_roll = roll
                st.session_state.student_data   = merged[merged["Roll_No"] == roll]
                st.session_state.page           = "admin_student"
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 ── ADMIN → STUDENT DETAIL VIEW
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "admin_student":

    student  = st.session_state.student_data
    roll     = int(student["Roll_No"].values[0])
    name     = student["Name"].values[0].strip()
    preds    = predict_student(roll, merged, models)
    recs     = get_recommendations(student, preds)
    sem3_avg = float(student["Sem3_Avg"].values[0])
    att_avg  = float(student["Attendance_Avg"].values[0])
    pred_avg = float(np.mean(list(preds.values())))
    grade, gcolor, gbg, gborder = get_grade(sem3_avg)
    pred_grade, pgcolor, *_ = get_grade(pred_avg)
    a_stat, a_color = att_status(att_avg)

    # ── Nav bar ───────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([4, 1, 1])
    with c1:
        st.markdown(f"""
        <div style='background:linear-gradient(90deg,#1a0533,#0d1526); border:1px solid #3b1f6b;
                    border-radius:12px; padding:13px 24px; margin-bottom:24px;'>
            <span style='color:#a78bfa; font-weight:700;'>🛡️ Admin Panel</span>
            <span style='color:#2d1b5e; margin:0 10px;'>|</span>
            <span style='color:#94a3b8; font-size:0.85rem;'>
                Viewing: <strong style='color:#e2e8f0;'>{name}</strong>
                &nbsp;·&nbsp; Roll No: {roll:02d}
            </span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        if st.button("← All Students", key="back_admin"):
            st.session_state.page = "admin"
            st.rerun()
    with c3:
        if st.button("⬅ Logout", key="admin_s_logout"):
            st.session_state.page         = "login"
            st.session_state.is_admin     = False
            st.session_state.student_data = None
            st.rerun()

    st.markdown(f"""
    <h2 style='color:#e2e8f0; font-size:1.4rem; font-weight:700; margin-bottom:4px;'>
        📋 {name} — Full Report
    </h2>
    <p style='color:#475569; font-size:0.82rem; margin-bottom:24px;'>
        Roll No: {roll:02d} · All 3 Semesters · Cluster Analysis · Sem 4 Prediction
    </p>
    """, unsafe_allow_html=True)

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='metric-row'>
        <div class='metric-card mc-blue'>
            <span class='metric-icon'>🎯</span>
            <div class='metric-value mv-blue'>{grade}</div>
            <div class='metric-label'>Current Grade</div>
        </div>
        <div class='metric-card mc-purple'>
            <span class='metric-icon'>🔮</span>
            <div class='metric-value mv-purple'>{pred_avg:.1f}</div>
            <div class='metric-label'>Predicted Sem 4</div>
        </div>
        <div class='metric-card mc-green'>
            <span class='metric-icon'>📅</span>
            <div class='metric-value mv-green'>{att_avg:.0f}%</div>
            <div class='metric-label'>Attendance · {a_stat}</div>
        </div>
        <div class='metric-card mc-orange'>
            <span class='metric-icon'>📊</span>
            <div class='metric-value mv-orange'>{sem3_avg:.1f}</div>
            <div class='metric-label'>Sem 3 Avg</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── All 3 semesters side by side ──────────────────────────────────────────
    st.markdown("<p class='section-title'>📚 Semester-wise Subject Marks</p>", unsafe_allow_html=True)

    sem_data = [
        ("Semester 1", [
            ("C Programming",  "C_Programming"),
            ("Maths I",        "Maths_I"),
            ("Digital Logic",  "Digital_Logic"),
        ], "Sem1_Attendance", "#3b82f6"),
        ("Semester 2", [
            ("Data Structures",   "Data_Structures"),
            ("Discrete Maths",    "Discrete_Maths"),
            ("Computer Org.",     "Computer_Organization"),
        ], "Sem2_Attendance", "#7c3aed"),
        ("Semester 3", [
            ("Algorithms",        "Algorithms"),
            ("Operating Systems", "Operating_Systems"),
            ("DBMS",              "DBMS"),
        ], "Sem3_Attendance", "#059669"),
    ]

    cols = st.columns(3)
    for col, (sem_name, subjects, att_col, col_accent) in zip(cols, sem_data):
        with col:
            att_val  = int(student[att_col].values[0])
            att_c    = "#34d399" if att_val >= 75 else ("#fbbf24" if att_val >= 65 else "#f87171")
            sem_avg  = float(np.mean([float(student[c].values[0]) for _, c in subjects]))
            g_s, gc_s, gbg_s, gbd_s = get_grade(sem_avg)

            rows_html = ""
            for subj_name, subj_col in subjects:
                sc = int(student[subj_col].values[0])
                g, gc, gbg2, gbd = get_grade(sc)
                status = "✓" if sc >= 40 else "✗"
                s_col  = "#34d399" if sc >= 40 else "#f87171"
                rows_html += (
                    f'<div style="display:flex; justify-content:space-between; align-items:center; '
                    f'padding:8px 0; border-bottom:1px solid #0d1829;">'
                    f'<div style="color:#94a3b8; font-size:.8rem;">{subj_name}</div>'
                    f'<div style="display:flex; align-items:center; gap:6px;">'
                    f'<span style="font-family:monospace; font-size:.85rem; font-weight:700; color:{gc};">{sc}</span>'
                    f'<span style="background:{gbg2}; color:{gc}; border:1px solid {gbd}; '
                    f'padding:1px 6px; border-radius:4px; font-size:.7rem; font-weight:700;">{g}</span>'
                    f'<span style="color:{s_col}; font-size:.75rem;">{status}</span>'
                    f'</div></div>'
                )

            st.markdown(f"""
            <div style='background:#0d111a; border:1px solid {col_accent}40;
                        border-top:3px solid {col_accent}; border-radius:13px; padding:16px 18px;'>
                <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>
                    <div style='font-size:.9rem; font-weight:700; color:#e2e8f0;'>{sem_name}</div>
                    <span style='background:{gbg_s}; color:{gc_s}; border:1px solid {gbd_s};
                                 padding:2px 9px; border-radius:5px; font-size:.75rem; font-weight:700;'>
                        Avg: {sem_avg:.1f} {g_s}
                    </span>
                </div>
                {rows_html}
                <div style='display:flex; justify-content:space-between; margin-top:10px;
                            padding-top:8px; border-top:1px solid #1e3a5f;'>
                    <div style='font-size:.75rem; color:#475569;'>Attendance</div>
                    <div style='font-size:.82rem; font-weight:700; color:{att_c};'>{att_val}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("<p class='section-title'>Visual Analysis</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(chart_attendance(student),      use_container_width=True)
    with c2: st.plotly_chart(chart_cluster_pie(student),     use_container_width=True)
    c3, c4 = st.columns(2)
    with c3: st.plotly_chart(chart_performance_trend(student), use_container_width=True)
    with c4: st.plotly_chart(chart_predicted(student, preds),  use_container_width=True)

    # ── Subject marks full table ───────────────────────────────────────────────
    st.markdown("<p class='section-title'>Subject Marks, Grades &amp; Status</p>", unsafe_allow_html=True)
    rows_html = ""
    for sname, col, clust, badge_cls, sem in SUBJECT_ROWS:
        sc = int(student[col].values[0])
        g, gc, gbg2, gbd = get_grade(sc)
        status_html = (
            '<span style="color:#34d399; font-size:0.78rem; font-weight:600;">✓ Pass</span>'
            if sc >= 40 else
            '<span style="color:#f87171; font-size:0.78rem; font-weight:600;">✗ Fail</span>'
        )
        rows_html += (
            f'<div class="subject-row">'
            f'<div class="subj-name">{sname}</div>'
            f'<div><span class="subj-cluster-badge {badge_cls}">{clust}</span></div>'
            f'<div style="color:#475569; font-size:0.8rem;">{sem}</div>'
            f'<div class="subj-score">{sc}</div>'
            f'<div style="display:flex; align-items:center; gap:8px;">'
            f'<span class="grade-badge" style="background:{gbg2}; color:{gc}; border:1px solid {gbd};">{g}</span>'
            f'{status_html}'
            f'</div>'
            f'</div>'
        )
    st.markdown(
        f'<div class="subject-table">'
        f'<div class="subject-table-header">'
        f'<div>Subject</div><div>Cluster</div><div>Semester</div><div>Score</div><div>Grade / Status</div>'
        f'</div>{rows_html}</div>',
        unsafe_allow_html=True
    )

    # ── PDF download ──────────────────────────────────────────────────────────
    st.markdown("<p class='section-title'>⬇️ Download Report as PDF</p>", unsafe_allow_html=True)
    pdf_buf = generate_pdf(student, preds, recs)
    st.download_button(
        label="📥  Download PDF Report",
        data=pdf_buf,
        file_name=f"SPP_Report_{name.replace(' ','_')}_Roll{roll:02d}.pdf",
        mime="application/pdf"
    )