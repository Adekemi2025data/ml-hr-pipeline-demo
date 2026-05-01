import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Load the HR drift datasets
reference = pd.read_csv("data/raw/WAFn-UseC-HR-Employee-Attrition.csv")
month1 = pd.read_csv("month1_hr_data.csv")
month2 = pd.read_csv("month2_hr_data.csv")
month3 = pd.read_csv("month3_hr_data.csv")

print("=" * 60)
print("HR DRIFT REPORT: Month 1 (expected: no drift)")
print("=" * 60)

report_month1 = Report(metrics=[DataDriftPreset()])
snapshot_month1 = report_month1.run(reference_data=reference, current_data=month1)
snapshot_month1.save_html("reports/hr_drift_month1.html")
print("Report saved to reports/hr_drift_month1.html")

print("\n" + "=" * 60)
print("HR DRIFT REPORT: Month 2 (expected: moderate drift)")
print("=" * 60)

report_month2 = Report(metrics=[DataDriftPreset()])
snapshot_month2 = report_month2.run(reference_data=reference, current_data=month2)
snapshot_month2.save_html("reports/hr_drift_month2.html")
print("Report saved to reports/hr_drift_month2.html")

print("\n" + "=" * 60)
print("HR DRIFT REPORT: Month 3 (expected: significant drift)")
print("=" * 60)

report_month3 = Report(metrics=[DataDriftPreset()])
snapshot_month3 = report_month3.run(reference_data=reference, current_data=month3)
snapshot_month3.save_html("reports/hr_drift_month3.html")
print("Report saved to reports/hr_drift_month3.html")

print("\nOpen the HTML files in your browser to explore the HR drift reports.")
