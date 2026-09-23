import json
import os

with open("ML_Project.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Cell 39 with Colab Form / Interactive UI compatible prediction code
interactive_cell_code = [
    "#@title 🔍 Interactive Insurance Claim Fraud Predictor { run: \"auto\" }\n",
    "# ============================================================\n",
    "# Test Any New Insurance Claim (Interactive Form Inputs)\n",
    "# ============================================================\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "from scipy import sparse\n",
    "\n",
    "# ------------------------------------------------------------\n",
    "# 1. User Form Inputs (Edit these values or use the Colab UI form)\n",
    "# ------------------------------------------------------------\n",
    "customer_age = 35 #@param {type:\"integer\"}\n",
    "claim_amount = 45000 #@param {type:\"number\"}\n",
    "incident_severity = \"Major Damage\" #@param [\"Minor Damage\", \"Total Loss\", \"Major Damage\", \"Trivial Damage\"]\n",
    "incident_type = \"Single Vehicle Collision\" #@param [\"Single Vehicle Collision\", \"Vehicle Theft\", \"Multi-vehicle Collision\", \"Parked Car\"]\n",
    "collision_type = \"Front Collision\" #@param [\"Front Collision\", \"Rear Collision\", \"Side Collision\", \"Unknown\"]\n",
    "authorities_contacted = \"Police\" #@param [\"Police\", \"Fire\", \"Ambulance\", \"Other\", \"Unknown\"]\n",
    "bodily_injuries = 1 #@param {type:\"slider\", min:0, max:5, step:1}\n",
    "witnesses = 0 #@param {type:\"slider\", min:0, max:5, step:1}\n",
    "police_report_available = \"YES\" #@param [\"YES\", \"NO\", \"Unknown\"]\n",
    "\n",
    "# ------------------------------------------------------------\n",
    "# 2. Build the Complete Feature Vector from Baseline Template\n",
    "# ------------------------------------------------------------\n",
    "# Start with median/mode template from X_test so all 33 features are present\n",
    "user_claim = X_test.iloc[0:1].copy()\n",
    "\n",
    "# Override with the user's specific inputs\n",
    "user_claim[\"customer_age\"] = customer_age\n",
    "user_claim[\"claim_amount\"] = claim_amount\n",
    "user_claim[\"incident_severity\"] = incident_severity\n",
    "user_claim[\"incident_type\"] = incident_type\n",
    "user_claim[\"collision_type\"] = collision_type\n",
    "user_claim[\"authorities_contacted\"] = authorities_contacted\n",
    "user_claim[\"bodily_injuries\"] = bodily_injuries\n",
    "user_claim[\"witnesses\"] = witnesses\n",
    "user_claim[\"police_report_available\"] = police_report_available\n",
    "\n",
    "# ------------------------------------------------------------\n",
    "# 3. Preprocess the User Claim\n",
    "# ------------------------------------------------------------\n",
    "processed_input = preprocessor.transform(user_claim)\n",
    "\n",
    "if sparse.issparse(processed_input):\n",
    "    input_dense = processed_input.toarray().astype(np.float32)\n",
    "else:\n",
    "    input_dense = np.asarray(processed_input).astype(np.float32)\n",
    "\n",
    "input_tensor = torch.tensor(input_dense, dtype=torch.float32).to(device)\n",
    "\n",
    "# ------------------------------------------------------------\n",
    "# 4. Predict Fraud Probability with PyTorch Student Model\n",
    "# ------------------------------------------------------------\n",
    "student.eval()\n",
    "with torch.no_grad():\n",
    "    logits = student(input_tensor)\n",
    "    fraud_prob = torch.sigmoid(logits).item()\n",
    "    routing_decision = route_claim(fraud_prob)\n",
    "\n",
    "# ------------------------------------------------------------\n",
    "# 5. Display Formatted Result Card\n",
    "# ------------------------------------------------------------\n",
    "print(\"\\n\" + \"=\" * 65)\n",
    "print(\"                CLAIM ANALYSIS & FRAUD ASSESSMENT\")\n",
    "print(\"=\" * 65)\n",
    "print(f\" Customer Age        : {customer_age} years\")\n",
    "print(f\" Claim Amount        : ${claim_amount:,.2f}\")\n",
    "print(f\" Incident Type       : {incident_type}\")\n",
    "print(f\" Incident Severity   : {incident_severity}\")\n",
    "print(f\" Collision Type      : {collision_type}\")\n",
    "print(f\" Authorities Notified: {authorities_contacted}\")\n",
    "print(f\" Witnesses / Report  : {witnesses} witness(es) | Police Report: {police_report_available}\")\n",
    "print(\"-\" * 65)\n",
    "print(f\" 🎯 Fraud Probability: {fraud_prob:.2%}\")\n",
    "print(f\" 🚦 Routing Decision : {routing_decision}\")\n",
    "print(\"=\" * 65)\n",
    "print(\"\\n--- MODEL EVALUATION METRICS ---\")\n",
    "print(\"Accuracy :\", teacher_accuracy)\n",
    "print(\"Precision:\", teacher_precision)\n",
    "print(\"Recall   :\", teacher_recall)\n",
    "print(\"F1 Score :\", teacher_f1)\n",
    "print(\"=\" * 65)\n"
]

nb['cells'][39]['source'] = interactive_cell_code

with open("ML_Project.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

downloads_file = os.path.expanduser(r"~\Downloads\ML_Project.ipynb")
if os.path.exists(downloads_file):
    with open(downloads_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

print("Interactive prediction cell added to ML_Project.ipynb in both locations!")
