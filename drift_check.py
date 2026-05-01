import pandas as pd
import sys
import json
from evidently import Report
from evidently.presets import DataDriftPreset

# Thresholds
DRIFT_SHARE_WARNING = 0.20
DRIFT_SHARE_CRITICAL = 0.40

def check_drift(reference_path, current_path):
    """Run drift analysis and return status."""
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)
    result = snapshot.dict()

    # Your Evidently version returns:
    # metrics[0]["value"] = {"count": <float>, "share": <float>}
    drift_block = result["metrics"][0]["value"]

    drifted = int(drift_block["count"])
    share = float(drift_block["share"])

    # Total features = number of columns in the dataset
    total = reference.shape[1]

    # Build result
    check_result = {
        "total_features": total,
        "drifted_features": drifted,
        "drift_share": round(share, 3),
        "dataset_drift": share >= 0.5,  # Evidently no longer provides this
        "status": "ok",
    }

    # Determine status
    if share >= DRIFT_SHARE_CRITICAL:
        check_result["status"] = "critical"
    elif share >= DRIFT_SHARE_WARNING:
        check_result["status"] = "warning"

    # Your Evidently version does NOT return per-column drift info
    # So we cannot list drifted feature names
    check_result["drifted_feature_names"] = []

    return check_result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python drift_check.py <reference_data.csv> <current_data.csv>")
        sys.exit(1)

    reference_path = sys.argv[1]
    current_path = sys.argv[2]

    print(f"Checking drift: {current_path} vs {reference_path}")
    print("=" * 60)

    result = check_drift(reference_path, current_path)

    print(f"Features drifted: {result['drifted_features']}/{result['total_features']} "
          f"({result['drift_share']*100:.1f}%)")
    print(f"Dataset drift:    {result['dataset_drift']}")
    print(f"Status:           {result['status'].upper()}")

    # Save result
    with open("reports/drift_check_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\nFull result saved to reports/drift_check_result.json")

    # Exit codes for CI/CD
    if result["status"] == "critical":
        print(f"\nCRITICAL: {result['drift_share']*100:.1f}% drifted "
              f"(threshold: {DRIFT_SHARE_CRITICAL*100:.0f}%)")
        sys.exit(1)

    elif result["status"] == "warning":
        print(f"\nWARNING: {result['drift_share']*100:.1f}% drifted "
              f"(threshold: {DRIFT_SHARE_WARNING*100:.0f}%)")
        sys.exit(0)

    else:
        print("\nAll clear. Feature distributions are stable.")
        sys.exit(0)
