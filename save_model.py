import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from scipy import sparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

print("1. Loading dataset...")
df = pd.read_excel("Insurance_Fraud_Dataset_100K_Realistic.xlsx")
df['authorities_contacted'] = df['authorities_contacted'].fillna('Unknown')
if "claim_id" in df.columns:
    df = df.drop(columns=["claim_id"])

y = df['previous_fraud_history']
X = df.drop('previous_fraud_history', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

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

print("2. Fitting preprocessor...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save preprocessor and sample template
joblib.dump(preprocessor, "preprocessor.pkl")
template_df = X_test.iloc[0:1].copy()
template_df.to_pickle("claim_template.pkl")

print("3. Training Teacher Model...")
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
teacher.fit(X_train_processed, y_train)
teacher_train_prob = teacher.predict_proba(X_train_processed)[:, 1]

print("4. Training Student Model (PyTorch)...")
if sparse.issparse(X_train_processed):
    X_train_dense = X_train_processed.toarray().astype(np.float32)
else:
    X_train_dense = np.asarray(X_train_processed).astype(np.float32)

y_train_np = y_train.to_numpy().astype(np.float32)
teacher_train_prob_np = teacher_train_prob.astype(np.float32)

train_dataset = TensorDataset(
    torch.tensor(X_train_dense, dtype=torch.float32),
    torch.tensor(y_train_np, dtype=torch.float32),
    torch.tensor(teacher_train_prob_np, dtype=torch.float32)
)
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

student = StudentModel(input_size)
optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
hard_loss_function = nn.BCEWithLogitsLoss()
temperature = 4.0
alpha = 0.5

student.train()
for epoch in range(15):
    for batch_X, batch_y, batch_teacher_prob in train_loader:
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

# Save teacher model and other artifacts
joblib.dump(teacher, "teacher_model.pkl")

# Save trained student model
torch.save({
    "model_state_dict": student.state_dict(),
    "input_size": input_size
}, "student_model.pt")

print("Saved model artifacts: 'teacher_model.pkl', 'student_model.pt', 'preprocessor.pkl', 'claim_template.pkl'!")
