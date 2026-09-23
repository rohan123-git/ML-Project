import json
import os

with open("ML_Project.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Cell 39
interactive_cell_code = [
    "# ============================================================\n",
    "# 🔍 MANUAL CUSTOMER INPUT & FRAUD PREDICTION\n",
    "# ============================================================\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "from scipy import sparse\n",
    "\n",
    "# ------------------------------------------------------------\n",
    "# 1. Enter Your Customer Details Here:\n",
    "# ------------------------------------------------------------\n",
    "customer_age = 45\n",
    "claim_amount = 62000.00\n",
    "auto_make = 'Toyota'\n",
    "auto_year = 2021\n",
    "incident_type = 'Single Vehicle Collision'\n",
    "incident_severity = 'Major Damage'    # 'Minor Damage', 'Total Loss', 'Major Damage', 'Trivial Damage'\n",
    "collision_type = 'Front Collision'     # 'Front Collision', 'Rear Collision', 'Side Collision', 'Unknown'\n",
    "authorities_contacted = 'Police'      # 'Police', 'Fire', 'Ambulance', 'None', 'Unknown'\n",
    "witnesses = 1\n",
    "police_report_available = 'YES'       # 'YES', 'NO', 'Unknown'\n",
    "bodily_injuries = 0\n",
    "\n",
    "# ------------------------------------------------------------\n",
    "# 2. Build Claim & Predict\n",
    "# ------------------------------------------------------------\n",
    "user_claim = X_test.iloc[0:1].copy()\n",
    "user_claim['customer_age'] = customer_age\n",
    "user_claim['claim_amount'] = claim_amount\n",
    "user_claim['auto_make'] = auto_make\n",
    "user_claim['auto_year'] = auto_year\n",
    "user_claim['incident_type'] = incident_type\n",
    "user_claim['incident_severity'] = incident_severity\n",
    "user_claim['collision_type'] = collision_type\n",
    "user_claim['authorities_contacted'] = authorities_contacted\n",
    "user_claim['witnesses'] = witnesses\n",
    "user_claim['police_report_available'] = police_report_available\n",
    "user_claim['bodily_injuries'] = bodily_injuries\n",
    "\n",
    "processed_input = preprocessor.transform(user_claim)\n",
    "if sparse.issparse(processed_input):\n",
    "    input_dense = processed_input.toarray().astype(np.float32)\n",
    "else:\n",
    "    input_dense = np.asarray(processed_input).astype(np.float32)\n",
    "\n",
    "input_tensor = torch.tensor(input_dense, dtype=torch.float32).to(device)\n",
    "\n",
    "student.eval()\n",
    "with torch.no_grad():\n",
    "    logits = student(input_tensor)\n",
    "    fraud_prob = torch.sigmoid(logits).item()\n",
    "    decision = route_claim(fraud_prob)\n",
    "\n",
    "# ------------------------------------------------------------\n",
    "# 3. Display Output\n",
    "# ------------------------------------------------------------\n",
    "print('=' * 68)\n",
    "print(f' Age: {customer_age} | Make: {auto_make} ({auto_year}) | Amount: ${claim_amount:,.2f}')\n",
    "print(f' Incident: {incident_type} ({incident_severity}) | Witnesses: {witnesses} | Police Report: {police_report_available}')\n",
    "print('-' * 68)\n",
    "print(f'-> Result: Fraud Probability = {fraud_prob:.2%} | Decision = {decision}')\n",
    "print('=' * 68)\n"
]

nb['cells'][39]['source'] = interactive_cell_code

with open("ML_Project.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

downloads_file = os.path.expanduser(r"~\Downloads\ML_Project.ipynb")
if os.path.exists(downloads_file):
    with open(downloads_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

print("Updated ML_Project.ipynb!")
