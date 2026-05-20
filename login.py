"""
pages/login.py — Login page (Roll Number + Password)
✅ FIXED: Correct password hints + session state checks
"""
import streamlit as st
from data_creation_model import STUDENT_PASSWORDS


def render(merged):
    """Render login page."""
    
    # ✅ FIXED: Initialize session state if needed
    if "login_error" not in st.session_state:
        st.session_state.login_error = ""

    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:

        # ── Header ────────────────────────────────────────────────────────────
        st.markdown("""
        <div style='padding-top:60px; text-align:center;'>
            <div style='font-size:3.2rem; margin-bottom:10px;'>🎓</div>
            <h1 style='font-size:1.6rem; font-weight:800;
                       background:linear-gradient(135deg,#60a5fa,#a78bfa,#34d399);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                       margin:0 0 6px;'>
                Student Performance<br>Prediction System
            </h1>
            <p style='color:#475569; font-size:.83rem; margin-bottom:32px;'>
                B.Tech CSE · Random Forest · 3 Cluster Analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Login card ────────────────────────────────────────────────────────
        st.markdown("""
        <div style='background:linear-gradient(145deg,#0d1526,#0a1020);
                    border:1px solid #1e3a5f; border-radius:18px; padding:32px 28px;'>
        """, unsafe_allow_html=True)

        roll_no  = st.number_input(
            "Roll Number",
            min_value=1, max_value=20, value=1, step=1
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password"
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Login button ──────────────────────────────────────────────────────
        col_btn, _ = st.columns([1, 1])
        with col_btn:
            login_btn = st.button("🚀  Login & View Report", use_container_width=True)

        # ── Error message ─────────────────────────────────────────────────────
        if st.session_state.login_error:
            st.markdown(
                f"<div class='error-msg'>❌ {st.session_state.login_error}</div>",
                unsafe_allow_html=True
            )

        # ── Validation logic ──────────────────────────────────────────────────
        if login_btn:
            try:
                student_row = merged[merged["Roll_No"] == roll_no]

                if student_row.empty:
                    st.session_state.login_error = "Roll number not found."

                elif not password.strip():
                    st.session_state.login_error = "Please enter your password."

                elif password.strip() != STUDENT_PASSWORDS.get(roll_no, ""):
                    st.session_state.login_error = f"Incorrect password for Roll No {roll_no}."

                else:
                    # ✅ Successful login
                    st.session_state.login_error  = ""
                    st.session_state.student_data = student_row
                    st.session_state.page         = "dashboard"
                    st.rerun()
                    
            except Exception as e:
                st.session_state.login_error = f"Login error: {str(e)}"

        # ── Demo hint ─────────────────────────────────────────────────────────
        # ✅ FIXED: Show actual passwords from STUDENT_PASSWORDS dict
        st.markdown("""
        <div style='margin-top:18px; background:rgba(59,130,246,0.06);
                    border:1px solid rgba(59,130,246,0.15); border-radius:10px;
                    padding:12px 16px;'>
            <div style='font-size:.7rem; color:#3b82f6; font-weight:700;
                        text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;'>
                Demo Credentials
            </div>
            <div style='font-size:.78rem; color:#64748b; line-height:1.8;'>
                Roll No <span style='color:#60a5fa; font-weight:600;'>1</span>
                &nbsp;→&nbsp; Password:
                <span style='color:#60a5fa; font-weight:600; font-family:monospace;'>
                    ankit1
                </span><br>
                Roll No <span style='color:#a78bfa; font-weight:600;'>2</span>
                &nbsp;→&nbsp; Password:
                <span style='color:#a78bfa; font-weight:600; font-family:monospace;'>
                    aryanshu2
                </span><br>
                Roll No <span style='color:#34d399; font-weight:600;'>3</span>
                &nbsp;→&nbsp; Password:
                <span style='color:#34d399; font-weight:600; font-family:monospace;'>
                    kritika3
                </span><br>
                <span style='color:#334155; font-size:.7rem;'>
                    Pattern: firstname + roll_number &nbsp;·&nbsp; All 20 students available
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='text-align:center; color:#334155; font-size:.73rem;
                    margin-top:20px; padding-bottom:36px;'>
            20 Students · 3 Semesters · 9 Subjects · AI Prediction
        </div>
        """, unsafe_allow_html=True)