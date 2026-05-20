# 🎓 Student Performance Prediction System
### B.Tech CSE | Random Forest | Cluster-Based Analysis

---

## 📁 Project Structure

```
student_performance_prediction/
│
├── app.py                  ← Main Streamlit Dashboard
├── data_creation.py        ← Generates student data (20 students × 3 sems)
├── prediction_model.py     ← Random Forest model + recommendations
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
└── output/
    ├── Student_Performance_Prediction.xlsx   ← Auto-generated data
    └── models.pkl                            ← Trained RF models
```

---

## 🚀 Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🧠 How It Works

### Data
- **20 students**, same across all 3 semesters
- **3 subjects per semester** (B.Tech CSE syllabus)
- **Attendance** tracked per semester

### Clusters
| Cluster | Sem 1 | Sem 2 | Sem 3 |
|---------|-------|-------|-------|
| 🖥️ Programming | C Programming | Data Structures | Algorithms |
| 📚 Theory | Digital Logic | Computer Org. | Operating Systems |
| 📐 Math/Logic | Maths I | Discrete Maths | DBMS |

### Prediction
- **Random Forest Regressor** (200 trees) trained per cluster
- Features: cluster scores across 3 sems + attendance trend + performance trend
- Predicts **Semester 4 score per cluster**

### Recommendations
- If predicted score < 55 OR current average < 55 → show YouTube videos + books
- Personalized per weak cluster

---

## 📊 Dashboard Features

1. **Class Overview** — all students table, distribution chart
2. **Student Search** — by Roll Number or Name
3. **Performance Report**
   - Semester trend line chart
   - Cluster radar chart
   - Attendance bar chart (with 75% line)
   - Subject heatmap
4. **Sem 4 Prediction** — cluster-wise predicted scores
5. **Recommendations** — YouTube links + book suggestions
6. **Score Breakdown** — detailed table with grades

---

## 🎓 Subjects Covered

**Semester 1:** C Programming, Maths I, Digital Logic  
**Semester 2:** Data Structures, Discrete Maths, Computer Organization  
**Semester 3:** Algorithms, Operating Systems, DBMS

---

*Built with Streamlit + Plotly + Scikit-learn*
