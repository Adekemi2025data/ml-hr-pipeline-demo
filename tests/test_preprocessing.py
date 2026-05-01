import pandas as pd
import numpy as np
import pytest
import sys

# Ensure src/ is on the Python path
sys.path.insert(0, "src")

from preprocessing import(
    fill_missing_with_median,
    normalize_column,
    encode_binary_column,
    create_age_bins,
    remove_outliers
)

# ───────────────────────────────────────────────
# Test: fill_missing_with_median
# ───────────────────────────────────────────────
def test_fill_missing_replaces_nulls():
    df = pd.DataFrame({
        "Age": [20.0, 30.0, np.nan, 40.0, 50.0]
    })

    result = fill_missing_with_median(df, ["Age"])

    assert result["Age"].isna().sum() == 0
    assert result["Age"].iloc[2] == 35.0  # median of [20,30,40,50]


def test_fill_missing_raises_for_missing_column():
    df = pd.DataFrame({"Age": [25, 30]})
    with pytest.raises(ValueError):
        fill_missing_with_median(df, ["NotAColumn"])


# ───────────────────────────────────────────────
# Test: normalize_column
# ───────────────────────────────────────────────
def test_normalize_min_max():
    df = pd.DataFrame({"MonthlyIncome": [1000, 2000, 3000]})
    result = normalize_column(df, "MonthlyIncome", method="min-max")

    assert result["MonthlyIncome"].min() == 0.0
    assert result["MonthlyIncome"].max() == 1.0


def test_normalize_zscore():
    df = pd.DataFrame({"DistanceFromHome": [10, 20, 30]})
    result = normalize_column(df, "DistanceFromHome", method="z-score")

    assert round(result["DistanceFromHome"].mean(), 6) == 0.0


# ───────────────────────────────────────────────
# Test: encode_binary_column
# ───────────────────────────────────────────────
def test_encode_binary_column():
    df = pd.DataFrame({"OverTime": ["Yes", "No", "Yes"]})
    result = encode_binary_column(df, "OverTime", positive_value="Yes")

    assert result["OverTime"].tolist() == [1, 0, 1]


def test_encode_binary_column_raises_for_nonbinary():
    df = pd.DataFrame({"Department": ["Sales", "HR", "R&D"]})
    with pytest.raises(ValueError):
        encode_binary_column(df, "Department", positive_value="Sales")


# ───────────────────────────────────────────────
# Test: create_age_bins
# ───────────────────────────────────────────────
def test_create_age_bins():
    df = pd.DataFrame({"Age": [22, 29, 45, 60]})
    result = create_age_bins(df, column="Age")

    assert "Age_bin" in result.columns
    assert result["Age_bin"].iloc[0] == "18-24"
    assert result["Age_bin"].iloc[2] == "35-49"


# ───────────────────────────────────────────────
# Test: remove_outliers
# ───────────────────────────────────────────────
def test_remove_outliers_iqr():
    df = pd.DataFrame({"YearsAtCompany": [1, 2, 3, 4, 100]})
    result = remove_outliers(df, "YearsAtCompany", method="iqr")

    # 100 should be removed as an outlier
    assert 100 not in result["YearsAtCompany"].values


def test_remove_outliers_zscore():
    df = pd.DataFrame({"MonthlyRate": [1000, 1100, 1200, 50000]})
    result = remove_outliers(df, "MonthlyRate", method="zscore", threshold=2)

    assert 50000 not in result["MonthlyRate"].values
    print("Result values:", result["MonthlyRate"].values)
