import os
import sys
import time
import json
import random
import datetime
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from scipy import sparse
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ==============================================================================
# 1. PYTORCH STUDENT MODEL ARCHITECTURE & ARTIFACT LOADER
# ==============================================================================
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

def load_artifacts():
    try:
        preprocessor = joblib.load("preprocessor.pkl")
        template_df = pd.read_pickle("claim_template.pkl")
        teacher_model = joblib.load("teacher_model.pkl")
        checkpoint = torch.load("student_model.pt", map_location=torch.device('cpu'), weights_only=False)
        student_model = StudentModel(checkpoint["input_size"])
        student_model.load_state_dict(checkpoint["model_state_dict"])
        student_model.eval()
        return preprocessor, template_df, teacher_model, student_model
    except Exception as e:
        print(f"[FATAL] Error loading model artifacts: {e}")
        sys.exit(1)

preprocessor, template_df, teacher_model, student_model = load_artifacts()

# ==============================================================================
# 2. INPUT NORMALIZATION & RISK INFERENCE ENGINE
# ==============================================================================
def normalize_claim_input(claim_dict: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(claim_dict)
    
    def clean_str(val, default=""):
        if val is None:
            return default
        s = str(val).strip()
        return s if s else default

    # Sex
    sex_raw = clean_str(normalized.get("insured_sex"), "MALE").upper()
    normalized["insured_sex"] = "FEMALE" if "FEM" in sex_raw else "MALE"

    # State
    state_raw = clean_str(normalized.get("policy_state"), "OH").upper()
    valid_states = ["OH", "CA", "TX", "NY", "FL", "IL", "IN", "PA", "MI"]
    normalized["policy_state"] = state_raw if state_raw in valid_states else "OH"

    # Auto Make
    make_raw = clean_str(normalized.get("auto_make"), "BMW").title()
    normalized["auto_make"] = make_raw if make_raw else "BMW"

    # Property Damage
    prop_raw = clean_str(normalized.get("property_damage"), "YES").upper()
    normalized["property_damage"] = "NO" if ("NO" in prop_raw or prop_raw == "0" or "FALSE" in prop_raw) else "YES"

    # Police Report Available
    pol_raw = clean_str(normalized.get("police_report_available"), "YES").upper()
    normalized["police_report_available"] = "NO" if ("NO" in pol_raw or pol_raw == "0" or "FALSE" in pol_raw) else "YES"

    # Incident Severity
    sev_raw = clean_str(normalized.get("incident_severity"), "Major").lower()
    if "total" in sev_raw or "loss" in sev_raw or "write" in sev_raw:
        normalized["incident_severity"] = "Total Loss"
    elif "minor" in sev_raw:
        normalized["incident_severity"] = "Minor"
    elif "trivial" in sev_raw or "scratch" in sev_raw:
        normalized["incident_severity"] = "Trivial"
    else:
        normalized["incident_severity"] = "Major"

    # Incident Type
    type_raw = clean_str(normalized.get("incident_type"), "Single Vehicle").lower()
    if "theft" in type_raw or "stolen" in type_raw:
        normalized["incident_type"] = "Vehicle Theft"
    elif "park" in type_raw:
        normalized["incident_type"] = "Parked Car"
    elif "multi" in type_raw or "many" in type_raw or "chain" in type_raw:
        normalized["incident_type"] = "Multi Vehicle"
    else:
        normalized["incident_type"] = "Single Vehicle"

    # Collision Type
    col_raw = clean_str(normalized.get("collision_type"), "Front").lower()
    if "rear" in col_raw or "back" in col_raw:
        normalized["collision_type"] = "Rear"
    elif "side" in col_raw or "door" in col_raw or "t-bone" in col_raw:
        normalized["collision_type"] = "Side"
    elif "unknown" in col_raw or "none" in col_raw or "n/a" in col_raw:
        normalized["collision_type"] = "Unknown"
    else:
        normalized["collision_type"] = "Front"

    # Authorities Contacted
    auth_raw = clean_str(normalized.get("authorities_contacted"), "Police").lower()
    if "fire" in auth_raw:
        normalized["authorities_contacted"] = "Fire"
    elif "amb" in auth_raw or "ems" in auth_raw or "medic" in auth_raw:
        normalized["authorities_contacted"] = "Ambulance"
    elif "none" in auth_raw or "no" in auth_raw:
        normalized["authorities_contacted"] = "Unknown"
    else:
        normalized["authorities_contacted"] = "Police"

    # Education Level
    edu_raw = clean_str(normalized.get("insured_education_level"), "College").lower()
    if "high" in edu_raw or "school" in edu_raw:
        normalized["insured_education_level"] = "High School"
    elif "assoc" in edu_raw:
        normalized["insured_education_level"] = "Associate"
    elif "master" in edu_raw or "md" in edu_raw or "jd" in edu_raw or "grad" in edu_raw:
        normalized["insured_education_level"] = "Masters"
    elif "phd" in edu_raw or "doc" in edu_raw:
        normalized["insured_education_level"] = "PhD"
    else:
        normalized["insured_education_level"] = "College"

    # Occupation
    occ_raw = clean_str(normalized.get("insured_occupation"), "Manager").lower()
    if "eng" in occ_raw or "tech" in occ_raw or "dev" in occ_raw or "it" in occ_raw or "soft" in occ_raw:
        normalized["insured_occupation"] = "Engineer"
    elif "doc" in occ_raw or "health" in occ_raw or "nurse" in occ_raw or "med" in occ_raw:
        normalized["insured_occupation"] = "Doctor"
    elif "law" in occ_raw or "legal" in occ_raw or "attorney" in occ_raw:
        normalized["insured_occupation"] = "Lawyer"
    elif "sale" in occ_raw or "market" in occ_raw or "biz" in occ_raw:
        normalized["insured_occupation"] = "Sales"
    elif "teach" in occ_raw or "prof" in occ_raw or "edu" in occ_raw:
        normalized["insured_occupation"] = "Teacher"
    elif "trade" in occ_raw or "craft" in occ_raw:
        normalized["insured_occupation"] = "Technician"
    else:
        normalized["insured_occupation"] = "Manager"

    # Safe Numeric casting
    try:
        normalized["customer_age"] = int(normalized.get("customer_age") or 35)
    except:
        normalized["customer_age"] = 35

    try:
        normalized["months_as_customer"] = int(normalized.get("months_as_customer") or 60)
    except:
        normalized["months_as_customer"] = 60

    try:
        normalized["previous_claims"] = int(normalized.get("previous_claims") or 0)
    except:
        normalized["previous_claims"] = 0

    try:
        normalized["claim_amount"] = float(normalized.get("claim_amount") or 45000.0)
    except:
        normalized["claim_amount"] = 45000.0

    try:
        normalized["policy_annual_premium"] = float(normalized.get("policy_annual_premium") or 1250.0)
    except:
        normalized["policy_annual_premium"] = 1250.0

    try:
        normalized["policy_deductable"] = int(normalized.get("policy_deductable") or 1000)
    except:
        normalized["policy_deductable"] = 1000

    try:
        normalized["auto_year"] = int(normalized.get("auto_year") or 2020)
    except:
        normalized["auto_year"] = 2020

    try:
        normalized["witnesses"] = int(normalized.get("witnesses") or 0)
    except:
        normalized["witnesses"] = 0

    try:
        normalized["bodily_injuries"] = int(normalized.get("bodily_injuries") or 0)
    except:
        normalized["bodily_injuries"] = 0

    return normalized

def evaluate_claim(claim_dict: Dict[str, Any]) -> Dict[str, Any]:
    normalized_dict = normalize_claim_input(claim_dict)
    
    sample = template_df.copy()
    for col, val in normalized_dict.items():
        if col in sample.columns:
            sample[col] = val

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

    # 1. XGBoost Teacher
    t_start = time.perf_counter()
    teacher_prob = float(teacher_model.predict_proba(processed)[:, 1][0])
    teacher_time = (time.perf_counter() - t_start) * 1000

    # 2. PyTorch Distilled Student
    s_start = time.perf_counter()
    tensor_in = torch.tensor(dense, dtype=torch.float32)
    with torch.no_grad():
        student_logits = student_model(tensor_in)
        student_prob = float(torch.sigmoid(student_logits).item())
    student_time = (time.perf_counter() - s_start) * 1000

    # 3. Forensic Risk Vector Points & Impacts
    risk_score_points = 0.0
    risk_factors = []
    feature_impacts = []

    sev = normalized_dict.get("incident_severity", "Major")
    if sev == "Total Loss":
        risk_score_points += 0.28
        risk_factors.append({
            "title": "⚠️ Total Asset Loss",
            "desc": "Severe total loss incident carrying maximum indemnity liability.",
            "tag": "danger"
        })
        feature_impacts.append({"name": "Total Loss Severity", "impact": 28})
    elif sev == "Major":
        risk_score_points += 0.14
        risk_factors.append({
            "title": "⚠️ Major Structural Damage",
            "desc": "Significant structural damage declared requiring high repair reimbursement.",
            "tag": "warning"
        })
        feature_impacts.append({"name": "Major Damage Severity", "impact": 14})
    elif sev in ["Minor", "Trivial"]:
        risk_score_points -= 0.10
        feature_impacts.append({"name": "Minor Damage Rating", "impact": -10})

    pol = normalized_dict.get("police_report_available", "YES")
    if str(pol).upper() == "NO":
        risk_score_points += 0.22
        risk_factors.append({
            "title": "📑 Missing Police Documentation",
            "desc": "No official police accident verification report on file for this incident.",
            "tag": "danger"
        })
        feature_impacts.append({"name": "Missing Police Report", "impact": 22})
    elif str(pol).upper() == "YES":
        risk_score_points -= 0.08
        feature_impacts.append({"name": "Police Report Corroborated", "impact": -8})

    wits = int(normalized_dict.get("witnesses", 0))
    if wits == 0:
        risk_score_points += 0.16
        risk_factors.append({
            "title": "👁️ Zero Corroborating Witnesses",
            "desc": "Unwitnessed incident lacking third-party or bystander validation.",
            "tag": "warning"
        })
        feature_impacts.append({"name": "Zero Independent Witnesses", "impact": 16})
    elif wits >= 2:
        risk_score_points -= 0.08
        feature_impacts.append({"name": "Witness Corroboration", "impact": -8})

    if tot >= 90000:
        risk_score_points += 0.24
        risk_factors.append({
            "title": "💰 Critical Financial Exposure",
            "desc": f"Claim amount (${tot:,.2f}) sits in top 5th percentile of portfolio exposure.",
            "tag": "danger"
        })
        feature_impacts.append({"name": "Critical Claim Exposure", "impact": 24})
    elif tot >= 55000:
        risk_score_points += 0.12
        risk_factors.append({
            "title": "💵 Elevated Claim Value",
            "desc": f"Total claimed amount (${tot:,.2f}) exceeds portfolio median.",
            "tag": "warning"
        })
        feature_impacts.append({"name": "Elevated Claim Value", "impact": 12})
    elif tot <= 8000:
        risk_score_points -= 0.12
        feature_impacts.append({"name": "Low Financial Exposure", "impact": -12})

    inc_t = normalized_dict.get("incident_type", "Single Vehicle")
    if inc_t == "Vehicle Theft":
        risk_score_points += 0.18
        risk_factors.append({
            "title": "🚗 Vehicle Theft Hazard",
            "desc": "Total unrecovered vehicle theft incident; requires key & immobilizer forensics.",
            "tag": "warning"
        })
        feature_impacts.append({"name": "Theft Hazard Category", "impact": 18})

    p_claims = int(normalized_dict.get("previous_claims", 0))
    if p_claims >= 3:
        risk_score_points += 0.20
        risk_factors.append({
            "title": "📈 Chronic Claim Frequency",
            "desc": f"Policyholder recorded {p_claims} prior claims within recent history.",
            "tag": "danger"
        })
        feature_impacts.append({"name": "Multiple Prior Claims", "impact": 20})
    elif p_claims >= 2:
        risk_score_points += 0.10
        risk_factors.append({
            "title": "📈 Elevated Claim History",
            "desc": f"Policyholder recorded {p_claims} prior claims.",
            "tag": "warning"
        })
        feature_impacts.append({"name": "Prior Claim Lodged", "impact": 10})
    elif p_claims == 0:
        risk_score_points -= 0.06
        feature_impacts.append({"name": "Clean Claim History", "impact": -6})

    tenure = int(normalized_dict.get("months_as_customer", 60))
    if tenure <= 12 and tot > 40000:
        risk_score_points += 0.14
        risk_factors.append({
            "title": "⏱️ Rapid First-Year Loss",
            "desc": f"New policyholder (tenure: {tenure} months) filing substantial claim.",
            "tag": "warning"
        })
        feature_impacts.append({"name": "Short Customer Tenure", "impact": 14})

    if not risk_factors:
        risk_factors.append({
            "title": "✅ Clean Risk Profile",
            "desc": "All primary forensic and policy indicators conform to legitimate claim distribution.",
            "tag": "clean"
        })

    # Composite Probability
    base_model = (0.60 * teacher_prob) + (0.40 * student_prob)
    calibrated_prob = (base_model * 2.0) + (risk_score_points * 0.75) + 0.08
    composite_prob = float(np.clip(calibrated_prob, 0.03, 0.98))

    # Routing Decision
    if composite_prob <= 0.32:
        routing_title = "Fast-Track Auto Settlement"
        risk_category = "Low Risk Profile"
        badge_style = "badge-low-risk"
        risk_color = "#34d399"
        action_summary = "Claim passes automated screening. Cleared for instant electronic ACH disbursement within 24 hours."
        escrow_status = "Approved & Escrow Released"
        action_code = "ACT-FT-200"
        sla = "< 24 Hours"
    elif composite_prob <= 0.68:
        routing_title = "Senior Adjuster Audit"
        risk_category = "Moderate Risk Review"
        badge_style = "badge-mid-risk"
        risk_color = "#fbbf24"
        action_summary = "Moderate risk indicators detected. Routed to tier-2 adjusters for physical damage and police verification."
        escrow_status = "Pending Adjuster Verification"
        action_code = "ACT-SA-302"
        sla = "48 - 72 Hours"
    else:
        routing_title = "Fraud Investigation Required"
        risk_category = "High Risk (Critical)"
        badge_style = "badge-high-risk"
        risk_color = "#f87171"
        action_summary = "Significant indicators of fraud or inflated loss. Escrow frozen; referred to Special Investigation Unit (SIU)."
        escrow_status = "Frozen / SIU Escalated"
        action_code = "ACT-SIU-911"
        sla = "Frozen Indefinitely"

    return {
        "teacher_prob": teacher_prob,
        "teacher_time": teacher_time,
        "student_prob": student_prob,
        "student_time": student_time,
        "composite_prob": composite_prob,
        "routing_title": routing_title,
        "risk_category": risk_category,
        "badge_style": badge_style,
        "risk_color": risk_color,
        "action_summary": action_summary,
        "escrow_status": escrow_status,
        "action_code": action_code,
        "sla": sla,
        "risk_factors": risk_factors,
        "feature_impacts": feature_impacts
    }

# ==============================================================================
# 3. FASTAPI SERVER & REST ENDPOINTS
# ==============================================================================
app = FastAPI(
    title="InsurAI Enterprise API",
    description="Automated Claim Risk Scoring & Fraud Detection Engine API",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClaimRequest(BaseModel):
    customer_age: Optional[int] = 38
    months_as_customer: Optional[int] = 60
    previous_claims: Optional[int] = 1
    insured_sex: Optional[str] = "MALE"
    insured_education_level: Optional[str] = "College"
    insured_occupation: Optional[str] = "Manager"
    policy_state: Optional[str] = "OH"
    claim_amount: Optional[float] = 45000.0
    policy_annual_premium: Optional[float] = 1250.0
    policy_deductable: Optional[int] = 1000
    auto_make: Optional[str] = "BMW"
    auto_year: Optional[int] = 2020
    property_damage: Optional[str] = "YES"
    incident_type: Optional[str] = "Single Vehicle"
    incident_severity: Optional[str] = "Major"
    collision_type: Optional[str] = "Front"
    authorities_contacted: Optional[str] = "Police"
    witnesses: Optional[int] = 0
    police_report_available: Optional[str] = "YES"
    bodily_injuries: Optional[int] = 0

@app.post("/api/predict")
def api_predict(claim: ClaimRequest):
    data = claim.dict()
    res = evaluate_claim(data)
    return {
        "status": "success",
        "timestamp": datetime.datetime.now().isoformat(),
        "input_parameters": data,
        "assessment": res
    }

@app.post("/api/batch-simulation")
def api_batch_simulation(size: int = 20):
    sim_results = []
    for _ in range(size):
        b_amt = round(random.uniform(2500.0, 140000.0), 2)
        b_make = random.choice(["Toyota", "Honda", "Tesla", "BMW", "Mercedes", "Audi", "Ford", "Hyundai", "Kia"])
        b_year = random.randint(2005, 2025)
        b_type = random.choice(["Single Vehicle", "Multi Vehicle", "Vehicle Theft", "Parked Car"])
        b_sev = random.choice(["Minor", "Major", "Total Loss", "Trivial"])
        b_rep = random.choice(["YES", "NO"])
        b_wit = random.randint(0, 4)
        b_inj = random.randint(0, 3)
        b_claims = random.randint(0, 4)
        b_age = random.randint(20, 75)

        p_load = {
            "customer_age": b_age, "claim_amount": b_amt, "auto_make": b_make, "auto_year": b_year,
            "incident_type": b_type, "incident_severity": b_sev, "police_report_available": b_rep,
            "witnesses": b_wit, "bodily_injuries": b_inj, "previous_claims": b_claims
        }
        res = evaluate_claim(p_load)
        sim_results.append({
            "claim_id": f"CLM-{random.randint(100000, 999999)}",
            "claim_amount": b_amt,
            "vehicle": f"{b_make} ({b_year})",
            "incident": f"{b_type} ({b_sev})",
            "police_report": b_rep,
            "witnesses": b_wit,
            "risk_score_pct": round(res['composite_prob'] * 100, 1),
            "risk_category": res['risk_category'],
            "routing_title": res['routing_title'],
            "escrow_status": res['escrow_status'],
            "raw_score": res['composite_prob']
        })
    
    total_exposure = sum(r["claim_amount"] for r in sim_results)
    fraud_escrow = sum(r["claim_amount"] for r in sim_results if r["raw_score"] > 0.68)
    fast_track_count = sum(1 for r in sim_results if r["raw_score"] <= 0.32)
    fraud_count = sum(1 for r in sim_results if r["raw_score"] > 0.68)

    return {
        "status": "success",
        "total_claims": size,
        "kpi": {
            "total_exposure": total_exposure,
            "fraud_escrow": fraud_escrow,
            "fast_track_count": fast_track_count,
            "fraud_count": fraud_count
        },
        "claims": sim_results
    }

# Ensure frontend directory exists
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    print("=" * 65)
    print(" InsurAI Enterprise Full-Stack Web Server Starting...")
    print(" Application URL : http://127.0.0.1:8000")
    print(" Interactive API : http://127.0.0.1:8000/docs")
    print("=" * 65)
    uvicorn.run(app, host="127.0.0.1", port=8000)
