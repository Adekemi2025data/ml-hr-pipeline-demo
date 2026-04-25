import pandas as pd
import numpy as np
import pytest
import sys

sys.path.insert(0, "src")

from preprocessing import validate_dataframe, clean_data, encode_categoricals, check_data_quality

# ───────────────────────────────────────────────────────────────
# HR SAMPLE DATA
# ───────────────────────────────────────────────────────────────

@pytest.fixture
def sample_hr_data():
    """Small dataset that mimics an HR attrition dataset structure."""
    return pd.DataFrame({
        "Age": [34.0, 29.0, np.nan, 41.0, 25.0, 30.0],
        "MonthlyIncome": [4500.0, 5200.0, 6100.0, np.nan, 3900.0, 4800.0],
        "YearsAtCompany": [5, 2, 7, 10, 1, 3],
        "JobRole": ["Sales Executive", "Research Scientist", "Sales Executive",
                    "Manager", "Laboratory Technician", "Manager"],
        "Department": ["Sales", "R&D", "Sales", "HR", "R&D", "HR"],
        "Gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
        "OverTime": ["Yes", "No", "Yes", "No", "Yes", "No"],
        "Attrition": [0, 1, 0, 1, 0, 0]
    })

# ───────────────────────────────────────────────────────────────
# validate_dataframe tests
# ───────────────────────────────────────────────────────────────

class TestValidateDataframe:

    def test_valid_dataframe_passes(self, sample_hr_data):
        result = validate_dataframe(
            sample_hr_data,
            required_columns=["Age", "Gender", "Attrition"],
            target_column="Attrition"
        )
        assert result is True

    def test_missing_column_raises(self, sample_hr_data):
        with pytest.raises(ValueError, match="Missing required HR columns"):
            validate_dataframe(
                sample_hr_data,
                required_columns=["Age", "Nonexistent"],
                target_column="Attrition"
            )

    def test_missing_target_raises(self, sample_hr_data):
        with pytest.raises(ValueError, match="Target column"):
            validate_dataframe(
                sample_hr_data,
                required_columns=["Age"],
                target_column="NonexistentTarget"
            )

    def test_empty_dataframe_raises(self):
        empty_df = pd.DataFrame({"Age": [], "Attrition": []})
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(empty_df, ["Age"], "Attrition")

# ───────────────────────────────────────────────────────────────
# clean_data tests
# ───────────────────────────────────────────────────────────────

class TestCleanData:

    def test_fills_numeric_nulls(self, sample_hr_data):
        result = clean_data(sample_hr_data, ["Age", "MonthlyIncome"], [])
        assert result["Age"].isna().sum() == 0
        assert result["MonthlyIncome"].isna().sum() == 0

    def test_does_not_modify_original(self, sample_hr_data):
        original_nulls = sample_hr_data["Age"].isna().sum()
        clean_data(sample_hr_data, ["Age"], [])
        assert sample_hr_data["Age"].isna().sum() == original_nulls

    def test_fills_with_median(self, sample_hr_data):
        result = clean_data(sample_hr_data, ["Age"], [])
        # Median of [34, 29, 41, 25, 30] = 30
        assert result["Age"].iloc[2] == 30.0

    def test_non_null_values_unchanged(self, sample_hr_data):
        result = clean_data(sample_hr_data, ["Age"], [])
        assert result["Age"].iloc[0] == 34.0
        assert result["Age"].iloc[1] == 29.0

# ───────────────────────────────────────────────────────────────
# encode_categoricals tests
# ───────────────────────────────────────────────────────────────

class TestEncodeCategoricals:

    def test_creates_dummy_columns(self, sample_hr_data):
        result = encode_categoricals(sample_hr_data, ["Gender"])
        assert "Gender" not in result.columns
        assert any("Gender" in col for col in result.columns)

    def test_drops_first_category(self, sample_hr_data):
        result = encode_categoricals(sample_hr_data, ["Gender"])
        gender_cols = [col for col in result.columns if "Gender" in col]
        assert len(gender_cols) == 1  # drop_first=True

    def test_preserves_row_count(self, sample_hr_data):
        result = encode_categoricals(sample_hr_data, ["Gender", "Department"])
        assert len(result) == len(sample_hr_data)

# ───────────────────────────────────────────────────────────────
# check_data_quality tests
# ───────────────────────────────────────────────────────────────

class TestDataQuality:

    def test_counts_nulls(self, sample_hr_data):
        report = check_data_quality(sample_hr_data, ["Age", "MonthlyIncome"])
        assert report["total_nulls"] == 2  # one in Age, one in MonthlyIncome

    def test_counts_rows(self, sample_hr_data):
        report = check_data_quality(sample_hr_data, ["Age"])
        assert report["total_rows"] == 6

    def test_reports_numeric_ranges(self, sample_hr_data):
        report = check_data_quality(sample_hr_data, ["YearsAtCompany"])
        assert report["YearsAtCompany_min"] == 1
        assert report["YearsAtCompany_max"] == 10
