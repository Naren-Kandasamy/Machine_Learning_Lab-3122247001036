import os
import json
import sys

BASE_DIR = "/home/nkandasamy/Desktop/M.Tech/III year/5th Sem/ML/Lab/Machine_Learning_Lab-3122247001036"

failures = []

def check(condition, message):
    if not condition:
        print(f"[FAIL] {message}")
        failures.append(message)
    else:
        print(f"[PASS] {message}")

print("==================================================")
print("VERIFYING EXPERIMENT 8 & 9 COMPLETE ARTIFACTS")
print("==================================================")

# 1. Dataset Verification
check(os.path.exists(os.path.join(BASE_DIR, "Datasets/UCI_HAR/train/X_train.txt")), "UCI HAR X_train.txt exists")
check(os.path.exists(os.path.join(BASE_DIR, "Datasets/UCI_HAR/activity_labels.txt")), "UCI HAR activity_labels.txt exists")
check(os.path.exists(os.path.join(BASE_DIR, "Datasets/English_Characters/english.csv")), "English Characters english.csv exists")
check(os.path.exists(os.path.join(BASE_DIR, "Datasets/English_Characters/Img")), "English Characters Img directory exists")

# 2. Vector Graphics Verification
ex8_figs = ['Elbow_Curve.eps', 'Silhouette_Curve.eps', 'Dendrogram.eps', 'Cluster_Scatter.eps', 'Clustering_Comparison.eps']
for fig in ex8_figs:
    p = os.path.join(BASE_DIR, "Ex8", fig)
    check(os.path.exists(p) and os.path.getsize(p) > 1000, f"Ex8 Figure: {fig} exists and non-empty")

ex9_figs = ['Loss_Convergence.eps', 'Confusion_Matrix.eps', 'ROC_Curves.eps']
for fig in ex9_figs:
    p = os.path.join(BASE_DIR, "Ex9", fig)
    check(os.path.exists(p) and os.path.getsize(p) > 1000, f"Ex9 Figure: {fig} exists and non-empty")

# 3. LaTeX PDF Reports Verification
p8 = os.path.join(BASE_DIR, "Ex8/Experiment_8_Report.pdf")
check(os.path.exists(p8) and os.path.getsize(p8) > 50000, f"Ex8 PDF Report exists ({os.path.getsize(p8) if os.path.exists(p8) else 0} bytes)")

p9 = os.path.join(BASE_DIR, "Ex9/Experiment_9_Report.pdf")
check(os.path.exists(p9) and os.path.getsize(p9) > 50000, f"Ex9 PDF Report exists ({os.path.getsize(p9) if os.path.exists(p9) else 0} bytes)")

# 4. Observation Documents Verification
obs8 = os.path.join(BASE_DIR, "Observation/Ex8_Observation.md")
check(os.path.exists(obs8) and "Table 1: K-Means Elbow Method Results" in open(obs8).read(), "Ex8 Observation contains Table 1")

obs9 = os.path.join(BASE_DIR, "Observation/Ex9_Observation.md")
check(os.path.exists(obs9) and "Table 2: A/B Performance Comparison" in open(obs9).read(), "Ex9 Observation contains Table 2")

# 5. Notebook Verification
nb_path = os.path.join(BASE_DIR, "ML_Lab.ipynb")
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
nb_text = "".join("".join(c.get('source', [])) for c in nb['cells'])
check("def run_experiment_8" in nb_text, "ML_Lab.ipynb contains run_experiment_8()")
check("def run_experiment_9" in nb_text, "ML_Lab.ipynb contains run_experiment_9()")
check("ex8_output = run_experiment_8()" in nb_text, "ML_Lab.ipynb contains ex8 master execution cell")
check("ex9_output = run_experiment_9()" in nb_text, "ML_Lab.ipynb contains ex9 master execution cell")

print("==================================================")
if failures:
    print(f"VERIFICATION FAILED: {len(failures)} checks failed.")
    sys.exit(1)
else:
    print("ALL VERIFICATION CHECKS PASSED PERFECTLY!")
    sys.exit(0)
