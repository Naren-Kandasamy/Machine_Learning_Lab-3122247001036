import json
import os
import sys

BASE_DIR = "/home/nkandasamy/Desktop/M.Tech/III year/5th Sem/ML/Lab/Machine_Learning_Lab-3122247001036"
GLOBAL_NB_PATH = os.path.join(BASE_DIR, "ML_Lab.ipynb")

with open(GLOBAL_NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Check if Experiment 3 is already in ML_Lab.ipynb
has_ex3 = any("Experiment 3" in "".join(c.get('source', [])) for c in nb['cells'])
if has_ex3:
    print("Experiment 3 already present in global notebook.")
    sys.exit(0)

# Import the cell generator functions from generate_all_notebooks
sys.path.append(BASE_DIR)
from generate_all_notebooks import create_ex3_nb, create_ex4_nb

ex3_cells = create_ex3_nb()
ex4_cells = create_ex4_nb()

# Find insertion point: right before "# Experiment 5"
insert_idx = -1
for i, cell in enumerate(nb['cells']):
    source_str = "".join(cell.get('source', []))
    if "# Experiment 5" in source_str:
        insert_idx = i
        break

if insert_idx == -1:
    print("Could not locate '# Experiment 5' cell. Appending at end.")
    insert_idx = len(nb['cells'])

new_cells = nb['cells'][:insert_idx] + ex3_cells + ex4_cells + nb['cells'][insert_idx:]
nb['cells'] = new_cells

with open(GLOBAL_NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Successfully inserted Experiment 3 & 4 into {GLOBAL_NB_PATH} at index {insert_idx}!")
print(f"Total cells in global notebook: {len(nb['cells'])}")
