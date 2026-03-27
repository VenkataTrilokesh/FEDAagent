from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "sample_data" / "iris.csv"
OUTPUT_DIR = PROJECT_ROOT / "example_artifacts"

cmd = [
    sys.executable,
    str(PROJECT_ROOT / "main.py"),
    "--file",
    str(DATASET),
    "--target",
    "species",
    "--output-dir",
    str(OUTPUT_DIR),
    "--export-html",
]

print("Running:", " ".join(str(c) for c in cmd))
subprocess.run(cmd, check=True)
print("Artifacts created in:", OUTPUT_DIR)
