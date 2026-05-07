"""
utilsProjectData.py
Utility functions for Grade Performance Analysis — CPSC 222 Spring 2026
Joe Silveira
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report


# ─────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────

def clean_category(row):
    """
    Normalize Canvas category labels into four standardized groups:
      - quizzes          : in-class quizzes
      - exams/tests      : midterm, final exam
      - big assignments  : data assignments, TDP projects, individual assignments
      - small assignments: mini assignments, participation
    """
    cat = str(row["Category"]).lower().strip()

    if "quiz" in cat:
        return "quizzes"

    if any(kw in cat for kw in ["exam", "midterm", "final"]):
        return "exams/tests"

    if any(kw in cat for kw in [
        "data assignment",
        "individual assignment",
        "design study",
        "prototype",
        "analytic evaluation",
        "usability study",
    ]):
        return "big assignments"

    return "small assignments"


def load_and_clean_grades(filepath="projectData.csv"):
    """
    Load the raw grades CSV, drop the duplicate header row Canvas sometimes
    inserts, coerce numeric columns, derive Percent, and add CleanCategory.
    Returns the cleaned DataFrame.
    """
    df = pd.read_csv(filepath)
    df = df[df["Score"] != "Score"].copy()
    df.reset_index(drop=True, inplace=True)
    df["CleanCategory"] = df.apply(clean_category, axis=1)
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df["OutOf"] = pd.to_numeric(df["OutOf"], errors="coerce")
    df["Percent"] = (df["Score"] / df["OutOf"]) * 100
    return df


def parse_date(s):
    """
    Parse Canvas date strings into datetime objects.
    Handles: 'Feb 1 11:59pm', 'Feb 1 11:59 pm', 'Jan 22 11am', 'Jan 22 11 am'
    Returns None if parsing fails or input is null.
    """
    if pd.isna(s):
        return None
    s = str(s).strip()
    formats = [
        "%b %d %I:%M%p",
        "%b %d %I:%M %p",
        "%b %d %I%p",
        "%b %d %I %p",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(f"{s} 2026", fmt + " %Y")
        except ValueError:
            pass
    return None


def add_timing_columns(df):
    """
    Parse Due and Submitted into datetimes, then derive:
      DayOfWeek : weekday name the assignment was due
      DaysEarly : days submitted before deadline (negative = late)
    Returns the modified DataFrame.
    """
    df = df.copy()
    df["DueDate"] = df["Due"].apply(parse_date)
    df["SubDate"] = df["Submitted"].apply(parse_date)
    df["DayOfWeek"] = df["DueDate"].apply(
        lambda x: x.strftime("%A") if pd.notna(x) else None
    )
    df["DaysEarly"] = (df["DueDate"] - df["SubDate"]).apply(
        lambda x: x.days if pd.notna(x) else None
    )
    return df


# ─────────────────────────────────────────────
# JOIN WITH CALENDAR TABLE
# ─────────────────────────────────────────────

def load_calendar(filepath="calendar.csv"):
    """
    Load the calendar reference table.
    Expected columns: DayOfWeek, IsWeekend (0=weekday, 1=weekend).
    """
    return pd.read_csv(filepath)


def join_with_calendar(df, cal):
    """
    Left-join grade data with calendar on DayOfWeek.
    Adds IsWeekend. Assignments with no due date get NaN.
    Returns merged DataFrame.
    """
    merged = df.merge(
        cal[["DayOfWeek", "IsWeekend"]].drop_duplicates(subset=["DayOfWeek"]),
        on="DayOfWeek",
        how="left"
    )
    return merged


# ─────────────────────────────────────────────
# EDA — SUMMARY STATISTICS
# ─────────────────────────────────────────────

def summary_statistics(df):
    """
    Print and return mean, median, std, min, max of Percent by CleanCategory.
    """
    graded = df.dropna(subset=["Percent"])
    summary = graded.groupby("CleanCategory")["Percent"].agg(
        Mean="mean", Median="median", Std="std", Min="min", Max="max"
    ).round(2)
    print(summary)
    return summary


# ─────────────────────────────────────────────
# EDA — VISUALIZATIONS
# ─────────────────────────────────────────────

def plot_grade_distribution_by_category(df):
    graded = df.dropna(subset=["Score", "OutOf"])
    plt.figure()
    graded.boxplot(column="Percent", by="CleanCategory")
    plt.ylabel("Percent")
    plt.title("Grade Distribution by Category")
    plt.suptitle("")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_avg_grade_by_category(df):
    graded = df.dropna(subset=["Score", "OutOf"])
    avg_by_cat = graded.groupby("CleanCategory")["Percent"].mean().sort_values()
    plt.figure()
    avg_by_cat.plot(kind="bar", color="steelblue")
    plt.title("Average Grade by Assignment Category")
    plt.ylabel("Average Percent")
    plt.xlabel("Category")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_avg_grade_by_day(df):
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    avg_by_day = (
        df.dropna(subset=["Percent", "DayOfWeek"])
        .groupby("DayOfWeek")["Percent"]
        .mean()
    )
    avg_by_day = avg_by_day.reindex([d for d in day_order if d in avg_by_day.index])
    plt.figure()
    avg_by_day.plot(kind="bar", color="steelblue")
    plt.title("Average Grade by Day of Week (Due Date)")
    plt.ylabel("Average Percent")
    plt.xlabel("Day of Week")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_avg_grade_weekend_vs_weekday(df):
    graded = df.dropna(subset=["Percent", "IsWeekend"])
    avg = graded.groupby("IsWeekend")["Percent"].mean()
    avg.index = avg.index.map({0: "Weekday", 1: "Weekend"})
    plt.figure()
    avg.plot(kind="bar", color=["steelblue", "salmon"])
    plt.title("Average Grade: Weekday vs Weekend Due Dates")
    plt.ylabel("Average Percent")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# HYPOTHESIS TESTS
# ─────────────────────────────────────────────

def test_mean_above_80(df):
    """
    One-sample t-test: H0: mean <= 80%, H1: mean > 80%.
    Returns (t_stat, p_value).
    """
    sample = df["Percent"].dropna()
    t_stat, p_value = stats.ttest_1samp(sample, popmean=80, alternative="greater")
    print(f"Sample mean: {sample.mean():.2f}%  (n={len(sample)})")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value:     {p_value:.4f}")
    if p_value < 0.05:
        print("-> Reject H0: Evidence that mean grade is significantly above 80%.")
    else:
        print("-> Fail to reject H0.")
    return t_stat, p_value


def test_quizzes_vs_non_quizzes(df):
    """
    Two-sample independent t-test.
    H0: mean grade on quizzes == mean grade on all other assignment types
    H1: quiz grades differ from non-quiz grades
    This tests whether in-class quizzes are scored differently from other work.
    Returns (t_stat, p_value).
    """
    graded = df.dropna(subset=["Percent", "CleanCategory"])
    quizzes = graded[graded["CleanCategory"] == "quizzes"]["Percent"]
    non_quiz = graded[graded["CleanCategory"] != "quizzes"]["Percent"]
    t_stat, p_value = stats.ttest_ind(quizzes, non_quiz, alternative="two-sided")
    print(f"Quiz mean:     {quizzes.mean():.2f}%  (n={len(quizzes)})")
    print(f"Non-quiz mean: {non_quiz.mean():.2f}%  (n={len(non_quiz)})")
    print(f"t-statistic:   {t_stat:.4f}")
    print(f"p-value:       {p_value:.4f}")
    if p_value < 0.05:
        print("-> Reject H0: Quiz grades are significantly different from other assignment grades.")
    else:
        print("-> Fail to reject H0: No significant difference detected.")
    return t_stat, p_value


# ─────────────────────────────────────────────
# MACHINE LEARNING
# ─────────────────────────────────────────────

def prepare_ml_data(df, threshold=85, test_size=0.2, random_state=42):
    """
    Prepare features and binary target for classification.
    HighScore = 1 if Percent >= threshold, else 0.
    Features: CategoryEncoded, DayEncoded, DaysEarly, OutOf.
    Returns X_train, X_test, y_train, y_test, features list, ml_df.
    """
    ml_df = df.dropna(
        subset=["Percent", "DayOfWeek", "DaysEarly", "OutOf", "CleanCategory"]
    ).copy()
    ml_df["HighScore"] = (ml_df["Percent"] >= threshold).astype(int)
    le_cat = LabelEncoder()
    le_day = LabelEncoder()
    ml_df["CategoryEncoded"] = le_cat.fit_transform(ml_df["CleanCategory"])
    ml_df["DayEncoded"] = le_day.fit_transform(ml_df["DayOfWeek"])
    features = ["CategoryEncoded", "DayEncoded", "DaysEarly", "OutOf"]
    X = ml_df[features]
    y = ml_df["HighScore"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, features, ml_df


def print_class_distribution(ml_df):
    counts = ml_df["HighScore"].value_counts().sort_index()
    total = len(ml_df)
    print("Class distribution:")
    for label, count in counts.items():
        name = "High (>=85%)" if label == 1 else "Low  (<85%)"
        print(f"  {name}: {count:3d}  ({count / total * 100:.1f}%)")


def run_knn(X_train, X_test, y_train, y_test, k_range=range(1, 15)):
    """
    Sweep k values, plot accuracy, then evaluate best k with classification_report.
    Returns the fitted best KNN model.
    """
    accuracies = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        accuracies.append(knn.score(X_test, y_test))
    plt.figure()
    plt.plot(list(k_range), accuracies, marker="o", color="steelblue")
    plt.title("KNN Accuracy vs. k")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()
    best_k = list(k_range)[accuracies.index(max(accuracies))]
    print(f"Best k: {best_k}  (accuracy: {max(accuracies):.2f})\n")
    knn_best = KNeighborsClassifier(n_neighbors=best_k)
    knn_best.fit(X_train, y_train)
    y_pred = knn_best.predict(X_test)
    print("KNN Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    return knn_best


def run_decision_tree(X_train, X_test, y_train, y_test, features, max_depth=4):
    """
    Train and evaluate a Decision Tree, then visualize it.
    Returns the fitted DecisionTreeClassifier.
    """
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print("Decision Tree Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    plt.figure(figsize=(16, 6))
    plot_tree(
        dt, feature_names=features,
        class_names=["Low", "High"],
        filled=True, rounded=True
    )
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.show()
    return dt
