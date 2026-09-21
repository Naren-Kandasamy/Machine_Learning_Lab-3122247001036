import os
import sys
import json
import time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

BASE_DIR = "/home/nkandasamy/Desktop/M.Tech/III year/5th Sem/ML/Lab/Machine_Learning_Lab-3122247001036"

# Strictly headless environment
os.environ["MPLBACKEND"] = "Agg"

notebooks_to_test = [
    "Ex1/Experiment_1.ipynb",
    "Ex2/Experiment_2.ipynb",
    "Ex3/Experiment_3.ipynb",
    "Ex4/Experiment_4.ipynb",
    "Ex5/Experiment_5.ipynb",
    "Ex6/Experiment_6.ipynb",
    "Ex7/Experiment_7.ipynb",
    "Ex8/Experiment_8.ipynb",
    "Ex9/Experiment_9.ipynb"
]

print("="*70)
print("STARTING AUTONOMOUS HEADLESS NOTEBOOK EXECUTION & VERIFICATION")
print("="*70)

failures = {}
successes = []

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

for nb_rel in notebooks_to_test:
    nb_full = os.path.join(BASE_DIR, nb_rel)
    print(f"\n[Executing] {nb_rel}...")
    t0 = time.time()
    try:
        with open(nb_full, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
            
        # Execute from repo root so relative paths like Datasets/... and Ex<N>/... resolve identically
        ep.preprocess(nb, {'metadata': {'path': BASE_DIR}})
        
        # Save back the executed notebook with populated cell outputs!
        with open(nb_full, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
            
        elapsed = time.time() - t0
        print(f"[SUCCESS] {nb_rel} executed in {elapsed:.2f}s with all outputs populated!")
        successes.append(nb_rel)
        
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[ERROR] {nb_rel} failed after {elapsed:.2f}s: {e}")
        failures[nb_rel] = str(e)

print("\n" + "="*70)
print("EXECUTION SUMMARY:")
print(f"Passed: {len(successes)} / {len(notebooks_to_test)}")
if failures:
    print(f"Failed: {len(failures)}")
    for nb_name, err in failures.items():
        print(f"  - {nb_name}: {err[:150]}...")
    sys.exit(1)
else:
    print("ALL 9 STANDALONE NOTEBOOKS EXECUTED FLAWLESSLY WITH CODE 0!")
    sys.exit(0)
