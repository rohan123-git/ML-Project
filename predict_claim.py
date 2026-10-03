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
    if probability <= 0.32:
        return "🟢 Fast-Track Payout (Low Risk - Auto Settlement)"
    elif probability <= 0.68:
        return "🟡 Standard Review (Moderate Risk - Adjuster Audit)"
    else:
        return "🔴 Fraud Investigation Required (High Risk - SIU Escrow)"

def load_inference_artifacts():
    preprocessor = joblib.load("preprocessor.pkl")
    template_df = pd.read_pickle("claim_template.pkl")
    teacher_model = joblib.load("teacher_model.pkl")
    
    checkpoint = torch.load("student_model.pt", map_location=torch.device('cpu'), weights_only=False)
    input_size = checkpoint["input_size"]
    student_model = StudentModel(input_size)
    student_model.load_state_dict(checkpoint["model_state_dict"])
    student_model.eval()
    
    return preprocessor, template_df, teacher_model, student_model

def normalize_claim_input(claim_dict):
    normalized = dict(claim_dict)
    
    sev_map = {
        "Minor Damage": "Minor", "Major Damage": "Major", "Total Loss": "Total Loss", "Trivial Damage": "Trivial",
        "Minor": "Minor", "Major": "Major", "Trivial": "Trivial"
    }
    if "incident_severity" in normalized:
        normalized["incident_severity"] = sev_map.get(normalized["incident_severity"], "Major")

    type_map = {
        "Single Vehicle Collision": "Single Vehicle", "Multi-vehicle Collision": "Multi Vehicle",
        "Vehicle Theft": "Vehicle Theft", "Parked Car": "Parked Car",
        "Single Vehicle": "Single Vehicle", "Multi Vehicle": "Multi Vehicle"
    }
    if "incident_type" in normalized:
        normalized["incident_type"] = type_map.get(normalized["incident_type"], "Single Vehicle")

    col_map = {
        "Front Collision": "Front", "Rear Collision": "Rear", "Side Collision": "Side", "Unknown": "Unknown",
        "Front": "Front", "Rear": "Rear", "Side": "Side"
    }
    if "collision_type" in normalized:
        normalized["collision_type"] = col_map.get(normalized["collision_type"], "Front")

    auth_map = {
        "Police": "Police", "Fire": "Fire", "Ambulance": "Ambulance",
        "None": "Unknown", "Other": "Unknown", "Unknown": "Unknown"
    }
    if "authorities_contacted" in normalized:
        normalized["authorities_contacted"] = auth_map.get(normalized["authorities_contacted"], "Police")

    return normalized

def predict_single_claim(preprocessor, template_df, teacher_model, student_model, user_inputs):
    normalized_dict = normalize_claim_input(user_inputs)
    
    sample = template_df.copy()
    for k, v in normalized_dict.items():
        if k in sample.columns:
            sample[k] = v
            
    tot = float(normalized_dict.get("claim_amount", 45000.0))
    inj_count = int(normalized_dict.get("bodily_injuries", 0))
    if inj_count > 0:
        sample["injury_claim"] = round(tot * 0.25, 2)
        sample["property_claim"] = round(tot * 0.15, 2)
        sample["vehicle_claim"] = round(tot * 0.60, 2)
    else:
        sample["injury_claim"] = 0.0
        sample["property_claim"] = round(tot * 0.20, 2)
        sample["vehicle_claim"] = round(tot * 0.80, 2)
    sample["claim_amount"] = tot

    processed = preprocessor.transform(sample)
    if sparse.issparse(processed):
        dense = processed.toarray().astype(np.float32)
    else:
        dense = np.asarray(processed).astype(np.float32)

    # Teacher & Student outputs
    t_prob = float(teacher_model.predict_proba(processed)[:, 1][0])
    
    tensor = torch.tensor(dense, dtype=torch.float32)
    with torch.no_grad():
        s_logits = student_model(tensor)
        s_prob = float(torch.sigmoid(s_logits).item())
        
    # Forensic calibration
    risk_points = 0.0
    sev = normalized_dict.get("incident_severity", "Major")
    if sev == "Total Loss":
        risk_points += 0.28
    elif sev == "Major":
        risk_points += 0.14
    elif sev in ["Minor", "Trivial"]:
        risk_points -= 0.10

    if normalized_dict.get("police_report_available") == "NO":
        risk_points += 0.22
    elif normalized_dict.get("police_report_available") == "YES":
        risk_points -= 0.08

    if int(normalized_dict.get("witnesses", 0)) == 0:
        risk_points += 0.16
    elif int(normalized_dict.get("witnesses", 0)) >= 2:
        risk_points -= 0.08

    if tot >= 90000:
        risk_points += 0.24
    elif tot >= 55000:
        risk_points += 0.12
    elif tot <= 8000:
        risk_points -= 0.12

    if normalized_dict.get("incident_type") == "Vehicle Theft":
        risk_points += 0.18

    p_claims = int(normalized_dict.get("previous_claims", 0))
    if p_claims >= 3:
        risk_points += 0.20
    elif p_claims >= 2:
        risk_points += 0.10
    elif p_claims == 0:
        risk_points -= 0.06

    base_model = (0.60 * t_prob) + (0.40 * s_prob)
    calibrated_prob = (base_model * 2.0) + (risk_points * 0.75) + 0.08
    composite_prob = float(np.clip(calibrated_prob, 0.03, 0.98))
    decision = route_claim(composite_prob)
        
    return composite_prob, decision, t_prob, s_prob

teacher_accuracy = 0.8398
teacher_precision = 0.0530
teacher_recall = 0.1317
teacher_f1 = 0.0756

def get_user_input_interactive():
    print("=" * 68)
    print("      ENTERPRISE INSURANCE CLAIM FRAUD SCREENING ENGINE")
    print("=" * 68)
    
    def prompt_float(prompt_text, default):
        val = input(f"{prompt_text} [default: {default}]: ").strip()
        if not val:
            return default
        try:
            return float(val)
        except ValueError:
            return default

    def prompt_str(prompt_text, options, default):
        opts_str = " / ".join(options)
        val = input(f"{prompt_text} ({opts_str}) [default: {default}]: ").strip()
        if not val or val not in options:
            return default
        return val

    age = prompt_float("1. Policyholder Age", 35)
    amount = prompt_float("2. Claim Amount ($)", 45000.0)
    severity = prompt_str("3. Incident Severity", ["Minor Damage", "Total Loss", "Major Damage", "Trivial Damage"], "Major Damage")
    incident_type = prompt_str("4. Incident Type", ["Single Vehicle Collision", "Vehicle Theft", "Multi-vehicle Collision", "Parked Car"], "Single Vehicle Collision")
    collision_type = prompt_str("5. Collision Point", ["Front Collision", "Rear Collision", "Side Collision", "Unknown"], "Front Collision")
    authorities = prompt_str("6. Authorities Notified", ["Police", "Fire", "Ambulance", "Other", "Unknown", "None"], "Police")
    witnesses = prompt_float("7. Number of Witnesses", 0)
    police_report = prompt_str("8. Police Report Available", ["YES", "NO", "Unknown"], "YES")
    injuries = prompt_float("9. Bodily Injuries Incurred", 0)
    prev_claims = prompt_float("10. Previous Claims Count", 0)

    return {
        "customer_age": age,
        "claim_amount": amount,
        "incident_severity": severity,
        "incident_type": incident_type,
        "collision_type": collision_type,
        "authorities_contacted": authorities,
        "witnesses": int(witnesses),
        "police_report_available": police_report,
        "bodily_injuries": int(injuries),
        "previous_claims": int(prev_claims)
    }

if __name__ == "__main__":
    preprocessor, template_df, teacher_model, student_model = load_inference_artifacts()
    user_inputs = get_user_input_interactive()
    
    prob, decision, t_prob, s_prob = predict_single_claim(
        preprocessor, template_df, teacher_model, student_model, user_inputs
    )
    
    print("\n" + "=" * 68)
    print("                     OFFICIAL CLAIM AUDIT REPORT")
    print("=" * 68)
    print(f" Customer Age        : {user_inputs['customer_age']:.0f} years")
    print(f" Claim Amount        : ${user_inputs['claim_amount']:,.2f}")
    print(f" Incident Type       : {user_inputs['incident_type']}")
    print(f" Incident Severity   : {user_inputs['incident_severity']}")
    print(f" Collision Type      : {user_inputs['collision_type']}")
    print(f" Authorities Notified: {user_inputs['authorities_contacted']}")
    print(f" Witnesses / Report  : {user_inputs['witnesses']} witness(es) | Police Report: {user_inputs['police_report_available']}")
    print("-" * 68)
    print(f" [AI INFERENCE] XGBoost Teacher Prob : {t_prob:.2%}")
    print(f" [AI INFERENCE] PyTorch Student Prob : {s_prob:.2%}")
    print(f" [RISK INDEX]   Composite Fraud Risk : {prob:.2%}")
    print(f" [DECISION]     Operational Action   : {decision}")
    print("=" * 68)
