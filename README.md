# 🎓 Student Performance Prediction System
### B.Tech CSE · Random Forest · Cluster-Based Analysis · Dark Theme Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?style=flat-square&logo=scikit-learn)![Plotly](https://img.shields.io/badge/Plotly-5.18+-purple?style=flat-square&logo=plotly)

---

## 📌 Overview

A machine learning web application built with **Streamlit** that predicts the **Semester 4 academic performance** of B.Tech CSE students using a **Random Forest Regressor**. The system groups subjects into 3 academic clusters, analyzes performance trends across 3 semesters, and provides personalized YouTube + book recommendations for weak areas.

The app supports two login types:
- 🎓 **Student Login** — view your own performance, predictions, and recommendations
- 🛡️ **Admin Login** — view all 20 students, class-wide charts, and individual reports

---

## 📁 Project Structure

```
student_performance_prediction/
│
├── app.py                  ← Main Streamlit app (all pages + routing)
├── data_creation.py        ← Generates synthetic student Excel data + passwords
├── prediction_model.py     ← Random Forest ML model + recommendations
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
└── output/                 ← Auto-created on first run
    ├── Student_Performance_Prediction.xlsx   ← Generated student data
    └── models.pkl                            ← Trained Random Forest models
```

---

## 🚀 Setup & Run

### Step 1 — Clone the repository
```bash
git clone https://github.com/bibaswan2905/student-performance-prediction.git
cd student-performance-prediction
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the app
```bash
streamlit run app.py
```

The app opens automatically in your browser at **`http://localhost:8501`**

> **Note:** On first run, the app auto-generates the Excel data file and trains the ML models. This takes about 10–15 seconds. Subsequent runs load from cache instantly.

---

## 🔐 Login System

### 🎓 Student Login
Each student logs in with their **Roll Number** and a **Password**.

| Roll No | Student Name | Password |
|---------|-------------|----------|
| 1 | ANKIT DAS | `ankit1` |
| 2 | ARYANSHU SAHA | `aryanshu2` |
| 3 | KRITIKA SORENG | `kritika3` |
| 4 | SNEHASHREE MOHANTY | `snehashree4` |
| 5 | BIBASWAN BEHERA | `bibaswan5` |
| 6 | CHOUDHARY ARUPA | `choudhary6` |
| 7 | KUMAR SATYA SANKALPA | `kumar7` |
| 8 | SHRUTI PASWAN | `shruti8` |
| 9 | CINCINNATI BISWAL | `cincinnati9` |
| 10 | AKANSHYA PATTNAIK | `akanshya10` |
| 11 | AMLAN DAS | `amlan11` |
| 12 | ABHISHEK DAS | `abhishek12` |
| 13 | SASWATI | `saswati13` |
| 14 | K LATISH | `klatish14` |
| 15 | SUBHASHREE BISWAL | `subhashree15` |
| 16 | PINAKA RUDRA | `pinaka16` |
| 17 | SASWAT PARIDA | `saswat17` |
| 18 | ANKIT BEHERA | `ankit18` |
| 19 | PRATEEK PATTNAIK | `prateek19` |
| 20 | AYUSH NAYAK | `ayush20` |

> Password pattern: **first name (lowercase) + roll number**

### 🛡️ Admin Login
| Username | Password |
|----------|----------|
| `admin` | `admin123` |

---

## 🧠 How the ML Works

### Data
- **20 students** tracked across **3 semesters**
- **3 subjects per semester** = 9 subjects total
- **Attendance** recorded per semester
- Data is synthetically generated with realistic score distributions using `numpy.random`

### Subject Clusters

The 9 subjects are grouped into 3 academic clusters based on subject relationships:

| Cluster | Semester 1 | Semester 2 | Semester 3 |
|---------|-----------|-----------|-----------|
| 🖥️ **Programming** | C Programming | Data Structures | Algorithms |
| 📚 **Theory** | Digital Logic | Computer Organization | Operating Systems |
| 📐 **Math / Logic** | Maths I | Discrete Maths | DBMS |

### Model
- **Algorithm:** Random Forest Regressor
- **Trees:** 200 estimators
- **Max Depth:** 5
- **One model trained per cluster** (3 models total)
- **Features used:**
  - Subject scores across all 3 semesters per cluster
  - Attendance per semester
  - Attendance trend (Sem3 − Sem1)
  - Performance trend (Sem3 avg − Sem1 avg)
- **Target:** Predicted Semester 4 score per cluster

### Recommendation Trigger
If **predicted score < 55** OR **current cluster average < 55** → show YouTube videos + book suggestions for that cluster

---

## 📊 Grade Scale

| Grade | Score Range |
|-------|------------|
| **O** | ≥ 90 |
| **A+** | ≥ 75 |
| **A** | ≥ 60 |
| **B** | ≥ 50 |
| **C** | ≥ 40 |
| **F** | < 40 |

---

## 🖥️ App Pages

### For Students

#### 1. Login Page
- Roll Number + Password authentication
- Error messages for wrong credentials

#### 2. Dashboard
- **Prediction banner** — Pass/Fail prediction with confidence %, risk level
- **4 metric cards** — Current Grade, Predicted Sem 4 score, Attendance, Sem 3 Average
- **Attendance alert** — Warning if below 75% minimum
- **Weak subjects panel** — Subjects scoring below 55
- **AI Recommendations panel** — Prioritized action items
- **Subject-wise performance bars** — Color-coded Strong / Average / Weak for all 9 subjects
- **4 Plotly charts:**
  - Attendance bar chart (with 75% threshold line)
  - Cluster radar/pie chart
  - Semester performance trend line
  - Predicted vs current comparison

#### 3. Detailed Report
- **Cluster-wise analysis cards** — Current avg, Sem 4 prediction, progress bar, alert status, insight text, subject breakdown bars
- **YouTube Resources** — 4 videos per weak cluster
- **Book Recommendations** — 4 books per weak cluster
- **PDF Download** — Professional 4-page white-theme PDF report

---

### For Admin

#### 4. Admin Panel (Homepage)
- **4 summary cards** — Total Students, Class Avg, Avg Attendance, At-Risk count
- **4 class-wide charts:**
  - Sem 3 Average bar chart for all 20 students
  - Cluster averages grouped bar chart (Programming / Theory / Math per student)
  - Attendance % bar chart with 75% threshold line
  - Semester trend — individual student lines + bold class average line
- **Semester filter** — View All / Sem 1 / Sem 2 / Sem 3
- **Full student results table** — scores, grades, attendance for all students
- **"View →" button** — opens any student's individual report

#### 5. Admin Student Detail View
- Individual student full report (same as student view)
- All 3 semester results displayed side-by-side
- Subject marks, grades, pass/fail per subject
- All 4 Plotly charts
- PDF download
- "← All Students" back button

---

## 📄 PDF Report

Generated using **ReportLab**. Professional white-theme, 4 pages:

| Page | Content |
|------|---------|
| 1 | Student overview — personal info, attendance summary, grade summary, Sem 1→3 score cards |
| 2 | Programming Cluster — current avg, Sem 4 prediction, subject breakdown, progress bar |
| 3 | Theory Cluster — same layout |
| 4 | Math/Logic Cluster + Recommendations (YouTube + Books) |

Download filename: `SPP_Report_StudentName_RollXX.pdf`

---

## 📦 Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
openpyxl>=3.1.0
plotly>=5.18.0
reportlab>=4.0.0
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🔧 Customization Guide

### Change student names or passwords
Edit `data_creation.py`:
```python
STUDENT_NAMES = ["YOUR NAME 1", "YOUR NAME 2", ...]

STUDENT_PASSWORDS = {
    1: "password1",
    2: "password2",
    ...
}
```
Then delete the `output/` folder and restart the app to regenerate data.

### Change admin credentials
Edit `app.py` and find:
```python
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"
```
Replace with your own credentials.

### Use real student data
Place your own Excel file at `output/Student_Performance_Prediction.xlsx` with these exact sheet names and columns:

**Semester_1 sheet:**
`Roll_No, Name, C_Programming, Maths_I, Digital_Logic, Sem1_Attendance`

**Semester_2 sheet:**
`Roll_No, Name, Data_Structures, Discrete_Maths, Computer_Organization, Sem2_Attendance`

**Semester_3 sheet:**
`Roll_No, Name, Algorithms, Operating_Systems, DBMS, Sem3_Attendance`

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Core language |
| **Streamlit** | Web app framework |
| **Scikit-learn** | Random Forest ML model |
| **Pandas + NumPy** | Data processing |
| **Plotly** | Interactive charts |
| **ReportLab** | PDF generation |
| **OpenPyXL** | Excel file read/write |

---
---

## 📝 License

This project is created for academic purposes as part of a B.Tech CSE project submission.

---
