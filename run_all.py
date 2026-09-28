"""Rebuild everything from the raw data, in order.

    py run_all.py

Step 2 (moving Stathead exports out of Downloads) is manual and already done;
its output is data/raw/stathead/. Everything else is rebuilt from data/raw/.
Takes about 10 minutes on a 20-core machine (steps 6a, 6 and 7 run in parallel).
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("Step 1: draft list", "src/step1_draft.py"),
    ("Step 3: dataset with yearly and cumulative AV and games", "src/step3_dataset.py"),
    ("Step 4: imputation evaluation (leave-one-class-out)", "src/step4_imputation_eval.py"),
    ("Step 5: multiple imputation of CUMYr5 for 2022-2025", "src/step5_impute.py"),
    ("Step 6a: simulation check of the Bayesian monotone regression", "src/step6a_simulation_check.py"),
    ("Step 6: bounded value chart", "src/step6_value_chart.py"),
    ("Step 7: comparison with other curves and the Jimmy Johnson chart", "src/step7_compare_models.py"),
    ("Figures", "src/make_figures.py"),
]

for title, script in STEPS:
    print(f"\n=== {title}  ({script}) ===", flush=True)
    start = time.time()
    path, *args = script.split()
    subprocess.run([sys.executable, str(ROOT / path), *args], check=True, cwd=ROOT)
    print(f"--- done in {time.time() - start:.0f}s")
print("\nAll steps finished.")
