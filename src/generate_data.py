import random
import csv
import os
import sys


def generate_hr_attrition_data(n_rows, output_path, seed=42):
    """Generate a synthetic HR employee attrition dataset."""
    random.seed(seed)

    header = [
        "EmployeeNumber", "Age", "Gender", "Department", "JobRole",
        "MonthlyIncome", "YearsAtCompany", "JobSatisfaction",
        "OverTime", "BusinessTravel", "DistanceFromHome",
        "NumCompaniesWorked", "Education", "MaritalStatus",
        "PerformanceRating", "Attrition"
    ]

    genders = ["Male", "Female"]
    departments = ["Sales", "Research & Development", "Human Resources"]
    job_roles = [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manager", "Healthcare Representative", "Human Resources",
        "Manufacturing Director", "Sales Representative"
    ]
    business_travel = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
    marital_status = ["Single", "Married", "Divorced"]

    rows = []

    for i in range(n_rows):
        age = random.randint(18, 60)
        years_at_company = random.randint(0, 40)
        monthly_income = random.randint(2000, 20000)
        job_satisfaction = random.randint(1, 4)
        distance = random.randint(1, 30)
        num_companies = random.randint(0, 10)
        education = random.randint(1, 5)
        performance = random.choice([3, 4])  # typical IBM dataset pattern
        overtime = random.choice(["Yes", "No"])

        # Base attrition probability
        attr_prob = 0.05

        # Increase attrition probability based on realistic HR patterns
        if overtime == "Yes":
            attr_prob += 0.15
        if job_satisfaction == 1:
            attr_prob += 0.20
        if years_at_company < 3:
            attr_prob += 0.10
        if distance > 20:
            attr_prob += 0.05
        if monthly_income < 4000:
            attr_prob += 0.10

        attrition = "Yes" if random.random() < attr_prob else "No"

        row = [
            i + 1,
            age,
            random.choice(genders),
            random.choice(departments),
            random.choice(job_roles),
            monthly_income,
            years_at_company,
            job_satisfaction,
            overtime,
            random.choice(business_travel),
            distance,
            num_companies,
            education,
            random.choice(marital_status),
            performance,
            attrition
        ]

        rows.append(row)

    # Save file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # Summary
    attr_count = sum(1 for r in rows if r[-1] == "Yes")
    print(f"Generated {n_rows} rows at {output_path}")
    print(f"Attrition rate: {attr_count/n_rows:.1%} ({attr_count} employees left)")


if __name__ == "__main__":
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/raw/hr_synthetic.csv"
    generate_hr_attrition_data(n_rows, output_path)
