import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("01_student_pass_fail.csv")
print(df.columns.tolist())

print("=" * 60)

print("STUDENT PASS/FAIL - EDA PROJECT")
print("=" * 60)

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nDATA TYPES:")
print(df.dtypes)

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nNUMBER OF DUPLICATE ROWS:")
print(df.duplicated().sum())

print("\nNEGATIVE STUDY HOURS:")
print(df[df["study_hours"] < 0])

print("\nATTENDANCE ABOVE 100:")
print(df[df["attendance"] > 100])

print("\nNEGATIVE ASSIGNMENTS:")
print(df[df["assignments_completed"] < 0])

print("\nPREVIOUS SCORE ABOVE 100:")
print(df[df["previous_score"] > 100])

plt.figure(figsize=(10, 6))

df[[
    "study_hours",
    "attendance",
    "assignments_completed",
    "previous_score"
]].boxplot()

plt.title("Boxplot of Student Data")
plt.xticks(rotation=45)
plt.ylabel("Values")
plt.show()

plt.figure(figsize=(8, 5))

plt.hist(df["study_hours"].dropna(), bins=20)

plt.xlabel("Study Hours")
plt.ylabel("Number of Students")
plt.title("Study Hours Distribution")

plt.show()

plt.figure(figsize=(6, 5))

df["result"].value_counts().plot(kind="bar")

plt.xlabel("result")
plt.ylabel("Number of Students")
plt.title("Pass vs Fail")

plt.show()

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="result",
    y="study_hours",
    data=df
)

plt.xlabel("result")
plt.ylabel("Study Hours")
plt.title("Study Hours vs Pass/Fail")

plt.show()

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="result",
    y="attendance",
    data=df
)

plt.xlabel("result")
plt.ylabel("Attendance")
plt.title("Attendance vs Pass/Fail")

plt.show()
plt.figure(figsize=(8, 6))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()

print("\n" + "=" * 60)
print("CLEANING DATA")
print("=" * 60)


df = df.drop_duplicates()


df.loc[
    df["study_hours"] < 0,
    "study_hours"
] = np.nan

df.loc[
    (df["attendance"] < 0) |
    (df["attendance"] > 100),
    "attendance"
] = np.nan

df.loc[
    df["assignments_completed"] < 0,
    "assignments_completed"
] = np.nan

df.loc[
    (df["previous_score"] < 0) |
    (df["previous_score"] > 100),
    "previous_score"
] = np.nan

df["study_hours"] = df["study_hours"].fillna(
    df["study_hours"].median()
)

df["attendance"] = df["attendance"].fillna(
    df["attendance"].median()
)

df["assignments_completed"] = df["assignments_completed"].fillna(
    df["assignments_completed"].median()
)

df["previous_score"] = df["previous_score"].fillna(
    df["previous_score"].median()
)


df["study_attendance_score"] = (
    df["study_hours"] * df["attendance"] / 100
)

df["assignment_completion_rate"] = (
    df["assignments_completed"] /
    df["assignments_completed"].max()
)

print("\nNEW FEATURES CREATED:")
print(
    df[
        [
            "study_attendance_score",
            "assignment_completion_rate"
        ]
    ].head()
)

print("\n" + "=" * 60)
print("CLEANED DATASET")
print("=" * 60)

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nFINAL SHAPE:")
print(df.shape)

print("\nFINAL MISSING VALUES:")
print(df.isnull().sum())

print("\nFINAL DUPLICATE ROWS:")
print(df.duplicated().sum())

X = df.drop("result", axis=1)

y = df["result"]

print("\n" + "=" * 60)
print("X AND Y")
print("=" * 60)

print("\nX (FEATURES):")
print(X.head())

print("\nY (TARGET):")
print(y.head())

print("\nX SHAPE:")
print(X.shape)

print("\nY SHAPE:")
print(y.shape)

df.to_csv(
    "cleaned_student_pass_fail.csv",
    index=False
)
print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nCleaned dataset saved as:")
print("cleaned_student_pass_fail.csv")