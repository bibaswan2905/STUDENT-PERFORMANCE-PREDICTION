import pandas as pd
import numpy as np
import os

STUDENT_NAMES = [
    "ANKIT DAS",              # Roll 1
    "ARYANSHU SAHA",          # Roll 2
    "KRITIKA SORENG",         # Roll 3
    "SNEHASHREE MOHANTY",     # Roll 4
    "BIBASWAN BEHERA",        # Roll 5
    "CHOUDHARY ARUPA",        # Roll 6
    "KUMAR SATYA SANKALPA",   # Roll 7
    "SHRUTI PASWAN",          # Roll 8
    "CINCINNATI BISWAL",      # Roll 9
    "AKANSHYA PATTNAIK",      # Roll 10
    "AMLAN DAS",              # Roll 11
    "ABHISHEK DAS",           # Roll 12
    "SASWATI ",               # Roll 13
    "K LATISH",               # Roll 14
    "SUBHASHREE BISWAL",      # Roll 15
    "PINAKA RUDRA",           # Roll 16
    "SASWAT PARIDA",          # Roll 17
    "AMIT BEHERA",           # Roll 18
    "PRATEEK PATTNAIK",       # Roll 19
    "AYUSH NAYAK",            # Roll 20
]

# ─────────────────────────────────────────────────────────────────────────────
# STUDENT PASSWORDS  —  Roll_No (int) : password (str)
# Password pattern = first name lowercase + roll number
# Example: Roll 1 → ankit1,  Roll 5 → bibaswan5
# To change any password just edit the value here.
# ─────────────────────────────────────────────────────────────────────────────
STUDENT_PASSWORDS = {
    1:  "ankit1",
    2:  "aryanshu2",
    3:  "kritika3",
    4:  "snehashree4",
    5:  "bibaswan5",
    6:  "choudhary6",
    7:  "kumar7",
    8:  "shruti8",
    9:  "cincinnati9",
    10: "akanshya10",
    11: "amlan11",
    12: "abhishek12",
    13: "saswati13",
    14: "klatish14",
    15: "subhashree15",
    16: "pinaka16",
    17: "saswat17",
    18: "ankit18",
    19: "prateek19",
    20: "ayush20",
}


def create_student_data():
    np.random.seed(42)

    sem1, sem2, sem3 = [], [], []

    for i in range(1, 21):
        # Sem 1 - base performance
        base = np.random.randint(45, 75)
        sem1.append({
            "Roll_No": i,
            "Name": STUDENT_NAMES[i-1],
            "C_Programming":         min(100, max(20, base + np.random.randint(-15, 20))),
            "Maths_I":               min(100, max(20, base + np.random.randint(-15, 20))),
            "Digital_Logic":         min(100, max(20, base + np.random.randint(-15, 20))),
            "Sem1_Attendance":       np.random.randint(55, 98)
        })

        # Sem 2 - slight variation from sem1
        base2 = base + np.random.randint(-5, 10)
        sem2.append({
            "Roll_No": i,
            "Name": STUDENT_NAMES[i-1],
            "Data_Structures":       min(100, max(20, base2 + np.random.randint(-15, 20))),
            "Discrete_Maths":        min(100, max(20, base2 + np.random.randint(-15, 20))),
            "Computer_Organization": min(100, max(20, base2 + np.random.randint(-15, 20))),
            "Sem2_Attendance":       np.random.randint(55, 98)
        })

        # Sem 3 - further variation
        base3 = base2 + np.random.randint(-5, 10)
        sem3.append({
            "Roll_No": i,
            "Name": STUDENT_NAMES[i-1],
            "Algorithms":            min(100, max(20, base3 + np.random.randint(-15, 20))),
            "Operating_Systems":     min(100, max(20, base3 + np.random.randint(-15, 20))),
            "DBMS":                  min(100, max(20, base3 + np.random.randint(-15, 20))),
            "Sem3_Attendance":       np.random.randint(55, 98)
        })

    df1 = pd.DataFrame(sem1)
    df2 = pd.DataFrame(sem2)
    df3 = pd.DataFrame(sem3)

    os.makedirs("output", exist_ok=True)
    file_path = "output/Student_Performance_Prediction.xlsx"

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        df1.to_excel(writer, sheet_name="Semester_1", index=False)
        df2.to_excel(writer, sheet_name="Semester_2", index=False)
        df3.to_excel(writer, sheet_name="Semester_3", index=False)

    print(f"✅ Data created → {file_path}")
    return file_path


if __name__ == "__main__":
    create_student_data()