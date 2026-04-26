HR Attrition Model & Drift Analysis Insights

This repository contains a complete HR Attrition Dataset designed for machine learning workflows, including:

Employee attrition prediction

Data preprocessing and validation

Model training and evaluation

Data drift simulation

Drift monitoring with Evidently AI

The dataset includes demographic, job‑related, performance, and satisfaction metrics commonly used in HR analytics and workforce modeling.
Numeric Features include :Age, DailyRate, DistanceFromHome, HourlyRate , MonthlyIncome , MonthlyRate, , NumCompaniesWorked, PercentSalaryHike, TotalWorkingYears, TrainingTimesLastYear, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager.
Categorical Features include, BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, OverTime
Target Variable: Attrition
0 = Employee stayed, 1 = Employee left
Dropped Columns
These columns are removed during preprocessing because they do not contribute to prediction: EmployeeCount , EmployeeNumber, Over18
Drift Overview Across Three Months
Month 1 — Minimal Drift (2/32 features, 6.2%)
•	Expected and healthy for a stable production environment
•	Only two features showed significant change
•	Dataset drift: False
Month 2 — Moderate Drift (7/32 features, 21.9%)
•	Noticeable increase in drifting features
•	Still within acceptable operational limits
•	Dataset drift: False
Month 3 — Highest Drift (8/32 features, 25%)
•	One quarter of features showed drift
•	Still not enough to trigger dataset level drift
•	Dataset drift: False
Across all months, drift increased gradually — a realistic pattern in HR environments where workforce composition and policies evolve over time.
Drift Scores
The drift threshold is 0.05.
•	Score > 0.05 → drift detected
•	Score < 0.05 → feature stable
This shows the monitoring system is working correctly, no dataset level drift , the drift is gradual, not sudden which is ideal for early detection
The Features to Watch Closely are 
•	StandardHours t
•	MonthlyRate 
•	PercentSalaryHike 
The Best Model: Logistic Regression
•	F1 Score: 0.4932 (best balance of precision & recall)
•	Accuracy: 87.41%
•	AUC ROC: 0.8048
 Training Script Insights
Themodel shows:
•	Accuracy: 74.9%
•	Precision: 17.95%
•	Recall: 11.93%
This indicates class imbalance, common in attrition datasets where most employees stay. High accuracy but low recall means the model struggles to identify employees who leave.

