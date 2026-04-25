import random
import csv
import os
import sys

def generate_hr_data(n_rows, output_path, seed=42):
    """Generate a synthetic HR attrition dataset."""
    random.seed(seed)

    header = [
        "Age", "Attrition", "BusinessTravel", "DailyRate", "Department",
        "DistanceFromHome", "Education", "EducationField", "EmployeeCount",
        "EmployeeNumber", "EnvironmentSatisfaction", "Gender", "HourlyRate",
        "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
        "MaritalStatus", "MonthlyIncome", "MonthlyRate",
        "NumCompaniesWorked", "Over18", "OverTime", "PercentSalaryHike",
        "PerformanceRating", "YearsAtCompany", "YearsInCurrentRole",
        "YearsSinceLastPromotion", "YearsWithCurrManager"
    ]

    # Example categorical mappings
    business_travel = [1, 2, 3]  # Rarely, Frequently, Travel_Regularly
    departments = [1, 2, 3]      # Sales, R&D, HR
    education_fields = [1, 2, 3, 4, 5, 6]
    job_roles = list(range(1, 10))
    marital_status = [1, 2, 3]   # Single, Married, Divorced

    rows = []
    for i in range(n_rows):

        # Attrition probability logic (example)
        attrition_prob = 0.10
        years = random.randint(0, 40)
        overtime = random.choice([1, 2])

        if overtime == 2:
            attrition_prob += 0.10
        if years < 3:
            attrition_prob += 0.15
        if random.randint(1, 5) == 5:
            attrition_prob += 0.05

        attrition = 1 if random.random() < attrition_prob else 0

        row = [
            random.randint(18, 60),          # Age
            attrition,                       # Attrition
            random.choice(business_travel),  # BusinessTravel
            random.randint(100, 1500),       # DailyRate
            random.choice(departments),      # Department
            random.randint(1, 30),           # DistanceFromHome
            random.randint(1, 5),            # Education
            random.choice(education_fields), # EducationField
            1,                                # EmployeeCount
            i + 1,                            # EmployeeNumber
            random.randint(1, 4),            # EnvironmentSatisfaction
            random.choice([1, 2]),           # Gender
            random.randint(30, 100),         # HourlyRate
            random.randint(1, 4),            # JobInvolvement
            random.randint(1, 5),            # JobLevel
            random.choice(job_roles),        # JobRole
            random.randint(1, 4),            # JobSatisfaction
            random.choice(marital_status),   # MaritalStatus
            random.randint(1000, 20000),     # MonthlyIncome
            random.randint(1000, 20000),     # MonthlyRate
            random.randint(0, 10),           # NumCompaniesWorked
            1,                                # Over18
            overtime,                         # OverTime
            random.randint(10, 25),          # PercentSalaryHike
            random.randint(1, 4),            # PerformanceRating
            years,                            # YearsAtCompany
            random.randint(0, years),         # YearsInCurrentRole
            random.randint(0, years),         # YearsSinceLastPromotion
            random.randint(0, years)          # YearsWithCurrManager
        ]

        rows.append(row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    attrition_count = sum(r[1] for r in rows)
    print(f"Generated {n_rows} rows at {output_path}")
    print(f"Attrition rate: {attrition_count/n_rows:.1%} ({attrition_count} employees left)")

if __name__ == "__main__":
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/raw/hr_data.csv"
    generate_hr_data(n_rows, output_path)
