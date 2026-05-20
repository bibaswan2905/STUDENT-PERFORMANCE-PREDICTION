import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os

# ─────────────────────────────────────────────
# CLUSTER DEFINITIONS
# ─────────────────────────────────────────────
# Programming Cluster:  C_Programming (S1) → Data_Structures (S2) → Algorithms (S3)
# Theory Cluster:       Digital_Logic (S1) → Computer_Organization (S2) → Operating_Systems (S3)
# Math/Logic Cluster:   Maths_I (S1) → Discrete_Maths (S2) → DBMS (S3)

CLUSTERS = {
    "Programming": {
        "Sem1": "C_Programming",
        "Sem2": "Data_Structures",
        "Sem3": "Algorithms"
    },
    "Theory": {
        "Sem1": "Digital_Logic",
        "Sem2": "Computer_Organization",
        "Sem3": "Operating_Systems"
    },
    "Math_Logic": {
        "Sem1": "Maths_I",
        "Sem2": "Discrete_Maths",
        "Sem3": "DBMS"
    }
}

ATTENDANCE_COLS = ["Sem1_Attendance", "Sem2_Attendance", "Sem3_Attendance"]

YOUTUBE_RECOMMENDATIONS = {
    "Programming": [
        {"title": "C Programming Full Course – FreeCodeCamp", "url": "https://www.youtube.com/watch?v=KJgsSFOSQv0"},
        {"title": "Data Structures – CS Dojo", "url": "https://www.youtube.com/watch?v=bum_19loj9A"},
        {"title": "Algorithms Full Course – Abdul Bari", "url": "https://www.youtube.com/watch?v=0IAPZzGSbME"},
        {"title": "DSA in Python – Jovian", "url": "https://www.youtube.com/watch?v=pkYVOmU3MgA"},
    ],
    "Theory": [
        {"title": "Digital Logic Design – Neso Academy", "url": "https://www.youtube.com/watch?v=M0mx8S05v60"},
        {"title": "Computer Organization – Gate Smashers", "url": "https://www.youtube.com/watch?v=Ol8D69VKX2k"},
        {"title": "Operating Systems – Neso Academy", "url": "https://www.youtube.com/watch?v=vBURTt97EkA"},
        {"title": "OS Concepts – Sudhakar Atchala", "url": "https://www.youtube.com/watch?v=mXw9ruZaxzQ"},
    ],
    "Math_Logic": [
        {"title": "Engineering Mathematics – GATE Wallah", "url": "https://www.youtube.com/watch?v=X5nEkj1kDJo"},
        {"title": "Discrete Mathematics – TrevTutor", "url": "https://www.youtube.com/watch?v=tyDKR4FG3Yw"},
        {"title": "DBMS Full Course – Gate Smashers", "url": "https://www.youtube.com/watch?v=kBdlM6hNDAE"},
        {"title": "SQL & DBMS – FreeCodeCamp", "url": "https://www.youtube.com/watch?v=HXV3zeQKqGY"},
    ]
}

BOOK_RECOMMENDATIONS = {
    "Programming": [
        {"title": "The C Programming Language", "author": "Kernighan & Ritchie"},
        {"title": "Introduction to Algorithms (CLRS)", "author": "Cormen et al."},
        {"title": "Data Structures & Algorithms in Python", "author": "Goodrich, Tamassia"},
        {"title": "Cracking the Coding Interview", "author": "Gayle Laakmann McDowell"},
    ],
    "Theory": [
        {"title": "Digital Design", "author": "M. Morris Mano"},
        {"title": "Computer Organization & Architecture", "author": "William Stallings"},
        {"title": "Operating System Concepts (Dinosaur Book)", "author": "Silberschatz et al."},
        {"title": "Modern Operating Systems", "author": "Andrew Tanenbaum"},
    ],
    "Math_Logic": [
        {"title": "Higher Engineering Mathematics", "author": "B.S. Grewal"},
        {"title": "Discrete Mathematics & Its Applications", "author": "Kenneth Rosen"},
        {"title": "Database System Concepts", "author": "Silberschatz, Korth & Sudarshan"},
        {"title": "Fundamentals of Database Systems", "author": "Elmasri & Navathe"},
    ]
}


def load_data():
    df1 = pd.read_excel("output/Student_Performance_Prediction.xlsx", sheet_name="Semester_1")
    df2 = pd.read_excel("output/Student_Performance_Prediction.xlsx", sheet_name="Semester_2")
    df3 = pd.read_excel("output/Student_Performance_Prediction.xlsx", sheet_name="Semester_3")
    return df1, df2, df3


def build_features(df1, df2, df3):
    """Merge all sems into one feature matrix per student."""
    merged = df1.merge(df2, on=["Roll_No", "Name"]).merge(df3, on=["Roll_No", "Name"])

    # Cluster scores per semester
    for sem, col_map in [("Sem1", {"Programming": "C_Programming", "Theory": "Digital_Logic", "Math_Logic": "Maths_I"}),
                          ("Sem2", {"Programming": "Data_Structures", "Theory": "Computer_Organization", "Math_Logic": "Discrete_Maths"}),
                          ("Sem3", {"Programming": "Algorithms", "Theory": "Operating_Systems", "Math_Logic": "DBMS"})]:
        for cluster, col in col_map.items():
            merged[f"{cluster}_{sem}"] = merged[col]

    # Overall cluster average across sems
    for cluster in ["Programming", "Theory", "Math_Logic"]:
        merged[f"{cluster}_Avg"] = merged[[f"{cluster}_Sem1", f"{cluster}_Sem2", f"{cluster}_Sem3"]].mean(axis=1)

    # Attendance trend
    merged["Attendance_Avg"] = merged[ATTENDANCE_COLS].mean(axis=1)
    merged["Attendance_Trend"] = merged["Sem3_Attendance"] - merged["Sem1_Attendance"]

    # Overall average per semester
    merged["Sem1_Avg"] = merged[["C_Programming", "Maths_I", "Digital_Logic"]].mean(axis=1)
    merged["Sem2_Avg"] = merged[["Data_Structures", "Discrete_Maths", "Computer_Organization"]].mean(axis=1)
    merged["Sem3_Avg"] = merged[["Algorithms", "Operating_Systems", "DBMS"]].mean(axis=1)

    # Performance trend
    merged["Performance_Trend"] = merged["Sem3_Avg"] - merged["Sem1_Avg"]

    return merged


def train_model(merged):
    """Train Random Forest to predict Sem4 performance per cluster."""
    feature_cols = [
        "Programming_Sem1", "Programming_Sem2", "Programming_Sem3",
        "Theory_Sem1", "Theory_Sem2", "Theory_Sem3",
        "Math_Logic_Sem1", "Math_Logic_Sem2", "Math_Logic_Sem3",
        "Sem1_Attendance", "Sem2_Attendance", "Sem3_Attendance",
        "Attendance_Trend", "Performance_Trend"
    ]

    models = {}
    for cluster in ["Programming", "Theory", "Math_Logic"]:
        # Simulate Sem4 target = weighted projection + noise (for training)
        np.random.seed(10)
        y = (
            0.2 * merged[f"{cluster}_Sem1"] +
            0.3 * merged[f"{cluster}_Sem2"] +
            0.5 * merged[f"{cluster}_Sem3"] +
            0.1 * (merged["Attendance_Avg"] - 75) +
            np.random.normal(0, 3, len(merged))
        ).clip(20, 100)

        X = merged[feature_cols]
        rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=5)
        rf.fit(X, y)
        models[cluster] = (rf, feature_cols)

    os.makedirs("output", exist_ok=True)
    with open("output/models.pkl", "wb") as f:
        pickle.dump(models, f)

    print("✅ Models trained and saved.")
    return models


def predict_student(roll_no, merged, models):
    """Predict Sem4 performance for a student."""
    student = merged[merged["Roll_No"] == roll_no]
    if student.empty:
        return None

    feature_cols = models["Programming"][1]
    X = student[feature_cols]

    predictions = {}
    for cluster, (model, cols) in models.items():
        pred = model.predict(X)[0]
        predictions[cluster] = round(pred, 2)

    return predictions


def get_recommendations(student_row, predictions):
    """Return YouTube + book recs for weak clusters."""
    recs = {}
    for cluster, pred_score in predictions.items():
        current_avg = student_row[f"{cluster}_Avg"].values[0]
        if pred_score < 55 or current_avg < 55:
            recs[cluster] = {
                "youtube": YOUTUBE_RECOMMENDATIONS[cluster],
                "books": BOOK_RECOMMENDATIONS[cluster],
                "current_avg": round(current_avg, 2),
                "predicted": pred_score
            }
    return recs


if __name__ == "__main__":
    df1, df2, df3 = load_data()
    merged = build_features(df1, df2, df3)
    models = train_model(merged)
    print("Feature engineering and model training complete!")
    print(merged[["Roll_No", "Name", "Programming_Avg", "Theory_Avg", "Math_Logic_Avg", "Sem3_Avg"]].head())
