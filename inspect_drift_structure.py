import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
import json

reference = pd.read_csv("reference_data.csv")
current = pd.read_csv("month1_data.csv")

report = Report(metrics=[DataDriftPreset()])
snapshot = report.run(reference_data=reference, current_data=current)

result = snapshot.dict()

print("\n=== TOP-LEVEL KEYS ===")
print(result.keys())

print("\n=== METRIC 0 KEYS ===")
print(result["metrics"][0].keys())

print("\n=== METRIC 0 VALUE ===")
print(json.dumps(result["metrics"][0], indent=2))
