import time
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier
from scipy import sparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

print("=" * 70)
print("1. LOADING DATASET")
print("=" * 70)
dataset_path = "Insurance_Fraud_Dataset_100K_Realistic.xlsx"
df = pd.read_excel(dataset_path)
print(f"Dataset shape: {df.shape}")

print("\n" + "=" * 70)
print("2. PREPROCESSING & CLEANING")
print("=" * 70)
# Handle missing values
df['authorities_contacted'] = df['authorities_contacted'].fillna('Unknown')
if "claim_id" in df.columns:
    df = df.drop(columns=["claim_id"])

# Split features & target
y = df['previous_fraud_history']
X = df.drop('previous_fraud_history', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} rows, {X_train.shape[1]} columns")
print(f"Testing set : {X_test.shape[0]} rows, {X_test.shape[1]} columns")

numeric_features = X_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print(f"Processed feature matrix shape: {X_train_processed.shape}")

print("\n" + "=" * 70)
print("3. TRAINING TEACHER MODEL (XGBoost)")
print("=" * 70)
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()
scale_pos_weight = negative / positive

teacher = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

start_t = time.time()
teacher.fit(X_train_processed, y_train)
print(f"Teacher training completed in {time.time() - start_t:.2f} seconds.")

teacher_pred = teacher.predict(X_test_processed)
teacher_prob = teacher.predict_proba(X_test_processed)[:, 1]

teacher_accuracy = accuracy_score(y_test, teacher_pred)
teacher_precision = precision_score(y_test, teacher_pred, zero_division=0)
teacher_recall = recall_score(y_test, teacher_pred, zero_division=0)
teacher_f1 = f1_score(y_test, teacher_pred, zero_division=0)
teacher_auc = roc_auc_score(y_test, teacher_prob)

print("\n--- TEACHER MODEL EVALUATION ---")
print(f"Accuracy : {teacher_accuracy:.4f}")
print(f"Precision: {teacher_precision:.4f}")
print(f"Recall   : {teacher_recall:.4f}")
print(f"F1 Score : {teacher_f1:.4f}")
print(f"ROC-AUC  : {teacher_auc:.4f}")

print("\n" + "=" * 70)
print("4. PREPARING KNOWLEDGE DISTILLATION (PyTorch)")
print("=" * 70)
teacher_train_prob = teacher.predict_proba(X_train_processed)[:, 1]

if sparse.issparse(X_train_processed):
    X_train_dense = X_train_processed.toarray().astype(np.float32)
    X_test_dense = X_test_processed.toarray().astype(np.float32)
else:
    X_train_dense = np.asarray(X_train_processed).astype(np.float32)
    X_test_dense = np.asarray(X_test_processed).astype(np.float32)

y_train_np = y_train.to_numpy().astype(np.float32)
y_test_np = y_test.to_numpy().astype(np.float32)
teacher_train_prob_np = teacher_train_prob.astype(np.float32)

X_train_tensor = torch.tensor(X_train_dense, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
teacher_train_prob_tensor = torch.tensor(teacher_train_prob_np, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor, teacher_train_prob_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

input_size = X_train_dense.shape[1]

class StudentModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
student = StudentModel(input_size).to(device)

temperature = 4.0
alpha = 0.5
optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
hard_loss_function = nn.BCEWithLogitsLoss()

print("\n" + "=" * 70)
print("5. TRAINING STUDENT MODEL VIA DISTILLATION")
print("=" * 70)
epochs = 15
student.train()

for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y, batch_teacher_prob in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        batch_teacher_prob = batch_teacher_prob.to(device)

        optimizer.zero_grad()
        student_logits = student(batch_X)
        hard_loss = hard_loss_function(student_logits, batch_y)

        eps = 1e-7
        teacher_logits = torch.log((batch_teacher_prob + eps) / (1 - batch_teacher_prob + eps))
        teacher_soft = torch.sigmoid(teacher_logits / temperature)
        student_soft = torch.sigmoid(student_logits / temperature)

        soft_loss = nn.functional.binary_cross_entropy(student_soft, teacher_soft)
        loss = alpha * hard_loss + (1 - alpha) * soft_loss * (temperature ** 2)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1:02d}/{epochs:02d} - Distillation Loss: {avg_loss:.4f}")

print("\n" + "=" * 70)
print("6. EVALUATING STUDENT MODEL")
print("=" * 70)
X_test_tensor = torch.tensor(X_test_dense, dtype=torch.float32).to(device)
student.eval()

with torch.no_grad():
    student_logits = student(X_test_tensor)
    student_prob = torch.sigmoid(student_logits).cpu().numpy()

student_pred = (student_prob >= 0.5).astype(int)

student_accuracy = accuracy_score(y_test_np, student_pred)
student_precision = precision_score(y_test_np, student_pred, zero_division=0)
student_recall = recall_score(y_test_np, student_pred, zero_division=0)
student_f1 = f1_score(y_test_np, student_pred, zero_division=0)
student_auc = roc_auc_score(y_test_np, student_prob)

print("\n--- STUDENT MODEL EVALUATION ---")
print(f"Accuracy : {student_accuracy:.4f}")
print(f"Precision: {student_precision:.4f}")
print(f"Recall   : {student_recall:.4f}")
print(f"F1 Score : {student_f1:.4f}")
print(f"ROC-AUC  : {student_auc:.4f}")

print("\n" + "=" * 70)
print("7. MODEL COMPARISON (Teacher vs Student)")
print("=" * 70)
comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
    "Teacher (XGBoost)": [teacher_accuracy, teacher_precision, teacher_recall, teacher_f1, teacher_auc],
    "Student (PyTorch)": [student_accuracy, student_precision, student_recall, student_f1, student_auc]
})
print(comparison.to_string(index=False))

def route_claim(prob):
    if prob <= 0.30:
        return "Fast-Track Payout"
    elif prob <= 0.70:
        return "Standard Review"
    else:
        return "Fraud Investigation"

print("\n" + "=" * 70)
print("8. AUTOMATIC CLAIM ROUTING - SAMPLE INFERENCE")
print("=" * 70)
sample_claim = X_test.iloc[0:1].copy()
sample_processed = preprocessor.transform(sample_claim)
if sparse.issparse(sample_processed):
    sample_dense = sample_processed.toarray().astype(np.float32)
else:
    sample_dense = np.asarray(sample_processed).astype(np.float32)

sample_tensor = torch.tensor(sample_dense, dtype=torch.float32).to(device)
student.eval()
with torch.no_grad():
    sample_prob = torch.sigmoid(student(sample_tensor)).item()
    decision = route_claim(sample_prob)

print(f"Customer Age      : {sample_claim['customer_age'].values[0]}")
print(f"Claim Amount      : ${sample_claim['claim_amount'].values[0]:,.2f}")
print(f"Incident Type     : {sample_claim['incident_type'].values[0]}")
print(f"Incident Severity : {sample_claim['incident_severity'].values[0]}")
print(f"Predicted Fraud Probability : {sample_prob:.2%}")
print(f"Routing Decision            : {decision}")
print("=" * 70)
print("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
