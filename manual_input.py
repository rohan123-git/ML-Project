import sys
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

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

def route_claim(probability):
    if probability <= 0.30:
        return "Fast-Track Payout (Low Risk)"
    elif probability <= 0.70:
        return "Standard Review (Medium Risk)"
    else:
        return "Fraud Investigation Required (High Risk)"

# Load saved model & preprocessor
preprocessor = joblib.load("preprocessor.pkl")
template_df = pd.read_pickle("claim_template.pkl")

checkpoint = torch.load("student_model.pt", map_location=torch.device('cpu'), weights_only=False)
model = StudentModel(checkpoint["input_size"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Pre-calculated Model Evaluation Metrics
teacher_accuracy = 0.8398
teacher_precision = 0.05301497369486038
teacher_recall = 0.13165829145728644
teacher_f1 = 0.07559145989613388

def predict_from_inputs(user_dict):
    sample = template_df.copy()
    for k, v in user_dict.items():
        if k in sample.columns:
            sample[k] = v
            
    processed = preprocessor.transform(sample)
    if sparse.issparse(processed):
        dense = processed.toarray().astype(np.float32)
    else:
        dense = np.asarray(processed).astype(np.float32)
        
    tensor = torch.tensor(dense, dtype=torch.float32)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).item()
        decision = route_claim(prob)
        
    return prob, decision

def get_input(prompt_text, default_val, cast_type=str):
    raw = input(f"{prompt_text}: ").strip()
    if not raw:
        return default_val
    try:
        return cast_type(raw)
    except Exception:
        return default_val

def main():
    print("=" * 70)
    print("       MANUAL INSURANCE CLAIM ENTRY & FRAUD PREDICTOR")
    print("=" * 70)
    print("Enter the customer and claim details:\n")
    
    while True:
        age = get_input("1. Customer Age", 35, int)
        claim_amount = get_input("2. Claim Amount ($)", 45000.0, float)
        auto_make = get_input("3. Auto Make (e.g. Toyota, Honda, Tesla, Audi, etc.)", "Toyota", str)
        auto_year = get_input("4. Vehicle Model Year (e.g. 2020)", 2020, int)
        incident_type = get_input("5. Incident Type (Single Vehicle Collision / Multi-vehicle Collision / Vehicle Theft / Parked Car)", "Single Vehicle Collision", str)
        incident_severity = get_input("6. Incident Severity (Minor Damage / Major Damage / Total Loss / Trivial Damage)", "Major Damage", str)
        collision_type = get_input("7. Collision Type (Front Collision / Rear Collision / Side Collision / Unknown)", "Front Collision", str)
        authorities = get_input("8. Authorities Contacted (Police / Fire / Ambulance / None / Unknown)", "Police", str)
        witnesses = get_input("9. Number of Witnesses", 0, int)
        police_report = get_input("10. Police Report Available (YES / NO / Unknown)", "YES", str)
        bodily_injuries = get_input("11. Number of Bodily Injuries (0-5)", 0, int)

        user_claim = {
            "customer_age": age,
            "claim_amount": claim_amount,
            "auto_make": auto_make,
            "auto_year": auto_year,
            "incident_type": incident_type,
            "incident_severity": incident_severity,
            "collision_type": collision_type,
            "authorities_contacted": authorities,
            "witnesses": witnesses,
            "police_report_available": police_report,
            "bodily_injuries": bodily_injuries
        }

        prob, decision = predict_from_inputs(user_claim)

        print("\n" + "=" * 70)
        print("                         PREDICTION RESULT")
        print("=" * 70)
        print(f"Age: {age} | Make: {auto_make} ({auto_year}) | Amount: ${claim_amount:,.2f}")
        print(f"Incident: {incident_type} ({incident_severity}) | Witnesses: {witnesses} | Police Report: {police_report}")
        print("-" * 70)
        print(f"-> Result: Fraud Probability = {prob:.2%} | Decision = {decision}")
        print("=" * 70)
        print("\n--- MODEL EVALUATION METRICS ---")
        print("Accuracy :", teacher_accuracy)
        print("Precision:", teacher_precision)
        print("Recall   :", teacher_recall)
        print("F1 Score :", teacher_f1)
        print("=" * 70)

        again = input("\nWould you like to enter another claim? (y/n): ").strip().lower()
        if again == 'n':
            print("\nThank you for using the Insurance Fraud Predictor!")
            break
        print("\n" + "-" * 70 + "\n")

if __name__ == "__main__":
    main()
