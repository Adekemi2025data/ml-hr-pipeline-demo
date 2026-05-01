import pandas as pd
import numpy as np

def load_and_prepare():
    """Load the HR attrition dataset and do basic cleaning."""
    path = "data/raw/WAFn-UseC-HR-Employee-Attrition.csv"
    df = pd.read_csv(path)

    df = df.drop(columns=["EmployeeNumber"], errors="ignore")

    df["Attrition"] = (
        df["Attrition"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "yes": 1, "y": 1, "1": 1, "true": 1,
            "no": 0, "n": 0, "0": 0, "false": 0
        })
    )
    df = df[df["Attrition"].isin([0, 1])]
    df["Attrition"] = df["Attrition"].astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def create_reference_and_production(df):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split = int(len(df) * 0.6)
    reference = df.iloc[:split].copy()
    remaining = df.iloc[split:].copy()

    batch_size = len(remaining) // 3
    month1 = remaining.iloc[:batch_size].copy()
    month2 = remaining.iloc[batch_size:batch_size*2].copy()
    month3 = remaining.iloc[batch_size*2:].copy()

    return reference, month1, month2, month3


def introduce_drift(month2, month3):
    month2.loc[month2.sample(frac=0.25, random_state=42).index, "OverTime"] = "Yes"
    month2["MonthlyIncome"] = month2["MonthlyIncome"] * np.random.uniform(1.02, 1.08)

    month3["MonthlyIncome"] = month3["MonthlyIncome"] * np.random.uniform(1.15, 1.35)
    month3.loc[month3.sample(frac=0.45, random_state=99).index, "OverTime"] = "Yes"

    dept_shift_idx = month3.sample(frac=0.20, random_state=10).index
    month3.loc[dept_shift_idx, "Department"] = "Sales"

    intern_count = int(len(month3) * 0.25)
    intern_indices = np.random.choice(month3.index, size=intern_count, replace=False)
    month3.loc[intern_indices, "Age"] = np.random.randint(18, 24, size=intern_count)

    new_hire_idx = month3.sample(frac=0.15, random_state=20).index
    month3.loc[new_hire_idx, "YearsAtCompany"] = np.random.randint(0, 2, size=len(new_hire_idx))

    return month2, month3


if __name__ == "__main__":
    print("Loading HR dataset...")
    df = load_and_prepare()
    print(f"Total rows: {len(df)}")

    print("\nSplitting into reference and production batches...")
    reference, month1, month2, month3 = create_reference_and_production(df)

    print("Introducing drift into months 2 and 3...")
    month2, month3 = introduce_drift(month2, month3)

    reference.to_csv("reference_data.csv", index=False)
    month1.to_csv("month1_data.csv", index=False)
    month2.to_csv("month2_data.csv", index=False)
    month3.to_csv("month3_data.csv", index=False)

    print("\nData saved. Ready for drift analysis.")

