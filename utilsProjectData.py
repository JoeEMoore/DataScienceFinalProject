import pandas as pd


def clean_category(row):
    name = str(row["Name"]).lower()
    category = str(row["Category"]).lower()

    try:
        outof = float(row["OutOf"])
    except:
        outof = None
    
    if "iq" in name or "quiz" in name:
        return "Quizzes"

    if "exam" in name or "midterm" in name or "final" in name:
        return "Exams/Tests"
    
    if (
        name.startswith("da") or 
        "individual assignment" in name or
        "tdp" in name or
        (outof is not None and outof >= 50)
    ):
        return "Big Assignments"
     # --- Small Assignments (default) ---
    return "Small Assignments"
# Convert to numeric



