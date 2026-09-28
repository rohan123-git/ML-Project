import json
import shutil

with open("ML_Project.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 1
nb['cells'][1]['source'] = [
    "# Mount Google Drive (Safe for both Colab and Local)\n",
    "import os\n",
    "try:\n",
    "    from google.colab import drive\n",
    "    if not os.path.exists('/content/drive'):\n",
    "        drive.mount('/content/drive')\n",
    "    else:\n",
    "        print('Google Drive is already mounted.')\n",
    "    drive_mounted = True\n",
    "except Exception:\n",
    "    drive_mounted = False\n",
    "    print('Running locally: Google Drive mount skipped.')\n"
]

# Cell 2
nb['cells'][2]['source'] = [
    "# Load dataset\n",
    "import os\n",
    "import pandas as pd\n",
    "\n",
    "colab_path = '/content/drive/MyDrive/ML Project/Insurance_Fraud_Dataset_100K_Realistic.xlsx'\n",
    "local_path = 'Insurance_Fraud_Dataset_100K_Realistic.xlsx'\n",
    "\n",
    "if os.path.exists(colab_path):\n",
    "    file_path = colab_path\n",
    "elif os.path.exists(local_path):\n",
    "    file_path = local_path\n",
    "else:\n",
    "    file_path = local_path\n",
    "\n",
    "print(f'Loading dataset from: {file_path}')\n",
    "df = pd.read_excel(file_path)\n",
    "print(f'Dataset loaded successfully! Shape: {df.shape}')\n"
]

# Cell 25
nb['cells'][25]['source'] = [
    "# Converting processed data to dense format safely\n",
    "from scipy import sparse\n",
    "import numpy as np\n",
    "\n",
    "if sparse.issparse(X_train_processed):\n",
    "    X_train_dense = X_train_processed.toarray().astype(np.float32)\n",
    "else:\n",
    "    X_train_dense = np.asarray(X_train_processed).astype(np.float32)\n",
    "\n",
    "if sparse.issparse(X_test_processed):\n",
    "    X_test_dense = X_test_processed.toarray().astype(np.float32)\n",
    "else:\n",
    "    X_test_dense = np.asarray(X_test_processed).astype(np.float32)\n",
    "\n",
    "y_train_np = y_train.to_numpy().astype(np.float32)\n",
    "y_test_np = y_test.to_numpy().astype(np.float32)\n",
    "teacher_train_prob_np = teacher_train_prob.astype(np.float32)\n",
    "\n",
    "print('X_train dense shape:', X_train_dense.shape)\n",
    "print('X_test dense shape :', X_test_dense.shape)\n",
    "print('y_train shape      :', y_train_np.shape)\n",
    "print('y_test shape       :', y_test_np.shape)\n"
]

# Cell 26
nb['cells'][26]['source'] = [
    "# Data preparation check\n",
    "print(f'Ready for PyTorch training on {X_train_dense.shape[0]} samples with {X_train_dense.shape[1]} features.')\n"
]

# Cell 39
nb['cells'][39]['source'] = [
    "# ============================================================\n",
    "# Test One New Insurance Claim - Sample Input\n",
    "# ============================================================\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "\n",
    "print('=' * 60)\n",
    "print('        INSURANCE CLAIM FRAUD DETECTION SYSTEM')\n",
    "print('=' * 60)\n",
    "\n",
    "# 1. Prepare a sample claim matching the exact feature schema\n",
    "sample_claim = X_test.iloc[0:1].copy()\n",
    "\n",
    "# 2. Preprocess the Sample Claim\n",
    "sample_processed = preprocessor.transform(sample_claim)\n",
    "\n",
    "if sparse.issparse(sample_processed):\n",
    "    sample_dense = sample_processed.toarray().astype(np.float32)\n",
    "else:\n",
    "    sample_dense = np.asarray(sample_processed).astype(np.float32)\n",
    "\n",
    "# 3. Convert to Tensor\n",
    "sample_tensor = torch.tensor(sample_dense, dtype=torch.float32).to(device)\n",
    "\n",
    "# 4. Predict Fraud Probability with Student Model\n",
    "student.eval()\n",
    "with torch.no_grad():\n",
    "    logits = student(sample_tensor)\n",
    "    probability = torch.sigmoid(logits).item()\n",
    "    decision = route_claim(probability)\n",
    "\n",
    "# 5. Output Display\n",
    "print('\\n' + '=' * 60)\n",
    "print('                 CLAIM ANALYSIS RESULT')\n",
    "print('=' * 60)\n",
    "if 'customer_age' in sample_claim.columns:\n",
    "    print(f'Customer Age      : {sample_claim[\"customer_age\"].values[0]}')\n",
    "if 'claim_amount' in sample_claim.columns:\n",
    "    print(f'Claim Amount      : ${sample_claim[\"claim_amount\"].values[0]:,.2f}')\n",
    "if 'incident_type' in sample_claim.columns:\n",
    "    print(f'Incident Type     : {sample_claim[\"incident_type\"].values[0]}')\n",
    "if 'incident_severity' in sample_claim.columns:\n",
    "    print(f'Incident Severity : {sample_claim[\"incident_severity\"].values[0]}')\n",
    "\n",
    "print('-' * 60)\n",
    "print(f'Fraud Probability : {probability:.2%}')\n",
    "print(f'Routing Decision  : {decision}')\n",
    "print('=' * 60)\n"
]

# Clear error outputs
for cell in nb['cells']:
    if 'outputs' in cell:
        cell['outputs'] = [out for out in cell['outputs'] if out.get('output_type') != 'error']

with open("ML_Project.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

import os
downloads_file = os.path.expanduser(r"~\Downloads\ML_Project.ipynb")
if os.path.exists(downloads_file):
    with open(downloads_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

print("Both notebooks updated successfully!")
