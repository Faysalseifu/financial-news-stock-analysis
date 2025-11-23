import os
import subprocess


def test_run_eda_and_outputs_exist():
    # Run the EDA script
    ret = subprocess.run(["python", "scripts/eda.py"], check=False)
    assert ret.returncode == 0, "EDA script did not complete successfully"

    expected = [
        "outputs/headline_length_stats.csv",
        "outputs/publisher_counts.csv",
        "outputs/daily_publication_counts.csv",
        "outputs/top_keywords.csv",
    ]
    for p in expected:
        assert os.path.exists(p), f"Expected output file missing: {p}"
