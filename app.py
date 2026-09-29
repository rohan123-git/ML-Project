import streamlit as st
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from scipy import sparse
import random
import time

# ==============================================================================
# 1. PAGE CONFIGURATION & ENTERPRISE DESIGN SYSTEM
# ==============================================================================
st.set_page_config(
    page_title="Insurance Fraud AI | Enterprise Decision Support",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS Styles
st.markdown("""
<style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .enterprise-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        color: #ffffff;
        padding: 1.75rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .enterprise-header h1 {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        color: #f8fafc;
    }
    
    .enterprise-header p {
        font-size: 0.95rem;
        margin: 0.35rem 0 0 0;
        color: #94a3b8;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.45rem 1rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 700;
        letter-spacing: 0.025em;
        text-transform: uppercase;
    }
    
    .badge-green {
        background-color: #064e3b;
        color: #6ee7b7;
        border: 1px solid #059669;
    }
    
    .badge-yellow {
        background-color: #713f12;
        color: #fde047;
        border: 1px solid #ca8a04;
    }
    
    .badge-red {
        background-color: #7f1d1d;
        color: #fca5a5;
        border: 1px solid #dc2626;
    }

    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .risk-score-high {
        color: #ef4444;
        font-weight: 800;
        font-size: 2.2rem;
    }
    .risk-score-mid {
        color: #eab308;
        font-weight: 800;
        font-size: 2.2rem;
    }
    .risk-score-low {
        color: #10b981;
        font-weight: 800;
        font-size: 2.2rem;
    }
    
    .report-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    .factor-item {
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        margin-bottom: 0.4rem;
        background: #1e293b;
        border-left: 4px solid #38bdf8;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. MODEL ARCHITECTURE & ARTIFACT LOADER
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

@st.cache_resource
def load_all_artifacts():
    try:
        preprocessor = joblib.load("preprocessor.pkl")
        template_df = pd.read_pickle("claim_template.pkl")
        
        # Load XGBoost Teacher
        teacher_model = joblib.load("teacher_model.pkl")
        
        # Load PyTorch Distilled Student
        checkpoint = torch.load("student_model.pt", map_location=torch.device('cpu'), weights_only=False)
        student_model = StudentModel(checkpoint["input_size"])
        student_model.load_state_dict(checkpoint["model_state_dict"])
        student_model.eval()
        
        return preprocessor, template_df, teacher_model, student_model
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        st.stop()

preprocessor, template_df, teacher_model, student_model = load_all_artifacts()

# ==============================================================================
# 3. ROUTING & RISK CALIBRATION LOGIC
# ==============================================================================
def compute_claim_assessment(claim_dict):
    """
    Constructs a full 33-feature record from the template and computes predictions
    from both XGBoost Teacher and Distilled PyTorch Student.
    """
    sample = template_df.copy()
    for col, val in claim_dict.items():
        if col in sample.columns:
            sample[col] = val

    # Automatically synchronize component claims if total amount is given
    tot = float(claim_dict.get("claim_amount", 45000.0))
    inj_count = int(claim_dict.get("bodily_injuries", 0))
    
    if inj_count > 0:
        sample["injury_claim"] = round(tot * 0.25, 2)
        sample["property_claim"] = round(tot * 0.15, 2)
        sample["vehicle_claim"] = round(tot * 0.60, 2)
    else:
        sample["injury_claim"] = 0.0
        sample["property_claim"] = round(tot * 0.20, 2)
        sample["vehicle_claim"] = round(tot * 0.80, 2)

    # Preprocessing
    processed = preprocessor.transform(sample)
    if sparse.issparse(processed):
        dense = processed.toarray().astype(np.float32)
    else:
        dense = np.asarray(processed).astype(np.float32)

    # 1. XGBoost Teacher Prediction
    t_start = time.perf_counter()
    teacher_prob = float(teacher_model.predict_proba(processed)[:, 1][0])
    teacher_time = (time.perf_counter() - t_start) * 1000

    # 2. PyTorch Distilled Student Prediction
    s_start = time.perf_counter()
    tensor_in = torch.tensor(dense, dtype=torch.float32)
    with torch.no_grad():
        student_logits = student_model(tensor_in)
        student_prob = float(torch.sigmoid(student_logits).item())
    student_time = (time.perf_counter() - s_start) * 1000

    # Composite Calibrated Risk Probability
    # Weights teacher probability (well-calibrated class weighting) & student representation
    composite_prob = (0.75 * teacher_prob) + (0.25 * student_prob)
    composite_prob = float(np.clip(composite_prob, 0.01, 0.99))

    # Determine Claim Routing according to Research Paper Policy
    if composite_prob <= 0.30:
        routing_title = "Fast-Track Payout"
        risk_category = "Low Risk"
        badge_style = "badge-green"
        action_summary = "Claim passes automated screening. Eligible for straight-through electronic settlement within 24 hours."
        escrow_status = "Released / Approved"
    elif composite_prob <= 0.70:
        routing_title = "Standard Review"
        risk_category = "Moderate Risk"
        badge_style = "badge-yellow"
        action_summary = "Claim exhibits moderate risk indicators. Routed to senior claims adjuster for manual documentation audit."
        escrow_status = "Pending Adjuster Verification"
    else:
        routing_title = "Fraud Investigation"
        risk_category = "High Risk (Critical)"
        badge_style = "badge-red"
        action_summary = "High probability of fraudulent or inflated claim characteristics. Escrow frozen and referred to Special Investigation Unit (SIU)."
        escrow_status = "Frozen / SIU Escalated"

    # Identify Key Risk Factors
    risk_factors = []
    if claim_dict.get("incident_severity") in ["Major Damage", "Total Loss"]:
        risk_factors.append(("⚠️ High Severity Incident", f"Severity declared as '{claim_dict.get('incident_severity')}' with substantial asset write-off risk."))
    if claim_dict.get("police_report_available") == "NO":
        risk_factors.append(("📑 Missing Police Documentation", "No official police report filed for the claimed incident."))
    if claim_dict.get("witnesses", 0) == 0:
        risk_factors.append(("👁️ Zero Independent Witnesses", "Absence of third-party witness corroboration."))
    if float(claim_dict.get("claim_amount", 0)) > 60000:
        risk_factors.append(("💰 Elevated Financial Exposure", f"Total claimed amount (${float(claim_dict.get('claim_amount', 0)):,.2f}) significantly exceeds portfolio median."))
    if claim_dict.get("incident_type") == "Vehicle Theft":
        risk_factors.append(("🚗 Total Vehicle Theft Incident", "High-frequency fraud category requiring key verification and physical tracking check."))
    if claim_dict.get("previous_claims", 0) >= 2:
        risk_factors.append(("📈 Elevated Claim Frequency", f"Policyholder recorded {claim_dict.get('previous_claims')} prior claim(s)."))

    if not risk_factors:
        risk_factors.append(("✅ Clean Risk Profile", "All primary indicators are consistent with standard legitimate claim patterns."))

    return {
        "teacher_prob": teacher_prob,
        "teacher_time": teacher_time,
        "student_prob": student_prob,
        "student_time": student_time,
        "composite_prob": composite_prob,
        "routing_title": routing_title,
        "risk_category": risk_category,
        "badge_style": badge_style,
        "action_summary": action_summary,
        "escrow_status": escrow_status,
        "risk_factors": risk_factors
    }

# ==============================================================================
# 4. SIDEBAR - INSTITUTIONAL CONTEXT
# ==============================================================================
with st.sidebar:
    st.markdown("### 🏛️ Research Context")
    st.info("""
    **Paper Title:**  
    *Insurance Claim Risk Classification Using XGBoost and Knowledge Distillation*  
    
    **Author:** Jatoth Akhil  
    **Department:** Information Technology  
    **Institution:** Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad
    """)
    st.markdown("---")
    st.markdown("### ⚙️ Engine Specifications")
    st.markdown("""
    - **Teacher Model:** XGBoost (`300 trees`, `scale_pos_weight=19.1`)
    - **Student Model:** Distilled PyTorch NN (`93-128-64-32-1`)
    - **Distillation Loss:** $\mathcal{L} = 0.5\mathcal{L}_{hard} + 0.5(4^2)\mathcal{L}_{soft}$
    - **Active Feature Space:** 93 Scaled/Encoded Features
    - **Target Label:** Historical Fraud Indicator
    """)
    st.markdown("---")
    st.caption("Enterprise Insurance Decision Support System v2.4")

# ==============================================================================
# 5. MAIN HEADER & TABS
# ==============================================================================
st.markdown("""
<div class="enterprise-header">
    <h1>🛡️ Enterprise Claim Risk & Fraud Routing AI</h1>
    <p>Real-Time Fraud Probability Estimation, Knowledge Distillation Auditing & Automated Tiered Routing Engine</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Interactive Claim Evaluator",
    "📊 Model Benchmark (Table 1)",
    "🎲 Batch & Random Stress Testing",
    "🔬 Knowledge Distillation Architecture"
])

# ------------------------------------------------------------------------------
# TAB 1: INTERACTIVE CLAIM EVALUATOR
# ------------------------------------------------------------------------------
with tab1:
    st.markdown("### 📝 Enter Insurance Claim Parameters")
    st.markdown("Select a real-world scenario preset or customize individual policy and incident parameters below.")
    
    # Preset Selector Bar
    p_col1, p_col2 = st.columns([3, 1])
    with p_col1:
        scenario = st.selectbox(
            "💡 Load Scenario Preset:",
            [
                "Custom Configuration",
                "Scenario 1: Minor Parking Scratch (Low Risk - Fast Payout)",
                "Scenario 2: Moderate Multi-Car Highway Collision (Standard Review)",
                "Scenario 3: Unwitnessed Luxury Vehicle Theft with Missing Police Report (High Risk - SIU Escalation)"
            ]
        )
    with p_col2:
        st.write("")
        st.write("")
        reset_btn = st.button("🔄 Reset to Defaults", use_container_width=True)

    # Apply Presets
    if scenario == "Scenario 1: Minor Parking Scratch (Low Risk - Fast Payout)":
        d_age, d_months, d_claims = 42, 84, 0
        d_amount, d_premium, d_deduct = 2800.0, 1100.0, 500
        d_make, d_year = "Toyota", 2021
        d_type, d_sev, d_col = "Parked Car", "Minor Damage", "Side Collision"
        d_auth, d_wit, d_rep, d_inj = "Police", 2, "YES", 0
    elif scenario == "Scenario 2: Moderate Multi-Car Highway Collision (Standard Review)":
        d_age, d_months, d_claims = 36, 48, 1
        d_amount, d_premium, d_deduct = 34000.0, 1350.0, 1000
        d_make, d_year = "Honda", 2018
        d_type, d_sev, d_col = "Multi-vehicle Collision", "Major Damage", "Rear Collision"
        d_auth, d_wit, d_rep, d_inj = "Police", 1, "YES", 1
    elif scenario == "Scenario 3: Unwitnessed Luxury Vehicle Theft with Missing Police Report (High Risk - SIU Escalation)":
        d_age, d_months, d_claims = 25, 12, 3
        d_amount, d_premium, d_deduct = 118000.0, 2400.0, 2000
        d_make, d_year = "Mercedes", 2023
        d_type, d_sev, d_col = "Vehicle Theft", "Total Loss", "Unknown"
        d_auth, d_wit, d_rep, d_inj = "None", 0, "NO", 0
    else:
        d_age, d_months, d_claims = 38, 60, 1
        d_amount, d_premium, d_deduct = 45000.0, 1250.0, 1000
        d_make, d_year = "BMW", 2020
        d_type, d_sev, d_col = "Single Vehicle Collision", "Major Damage", "Front Collision"
        d_auth, d_wit, d_rep, d_inj = "Police", 0, "YES", 0

    with st.form("interactive_claim_form"):
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("#### 👤 1. Policyholder Profile")
            age = st.number_input("Policyholder Age", min_value=18, max_value=95, value=d_age)
            months_customer = st.number_input("Tenure with Insurer (Months)", min_value=0, max_value=600, value=d_months)
            previous_claims = st.number_input("Prior Claims Count", min_value=0, max_value=15, value=d_claims)
            insured_sex = st.selectbox("Insured Sex", ["MALE", "FEMALE"])
            education_level = st.selectbox("Education Level", ["Bachelor", "College", "High School", "Associate", "JD", "MD", "PhD"], index=0)
            occupation = st.selectbox("Occupation", ["Executive", "Doctor", "Engineer", "Lawyer", "Sales", "Teacher", "Technician", "Other"])
            policy_state = st.selectbox("Policy State", ["OH", "IL", "IN", "CA", "TX", "NY", "FL"])

        with col_b:
            st.markdown("#### 🚗 2. Vehicle & Financial Terms")
            claim_amount = st.number_input("Total Claimed Amount ($)", min_value=500.0, max_value=500000.0, value=float(d_amount), step=500.0)
            policy_annual_premium = st.number_input("Annual Premium ($)", min_value=200.0, max_value=15000.0, value=float(d_premium), step=50.0)
            policy_deductable = st.selectbox("Policy Deductible ($)", [500, 1000, 2000], index=[500, 1000, 2000].index(d_deduct))
            auto_make = st.selectbox("Vehicle Make", ["Toyota", "Honda", "Tesla", "BMW", "Mercedes", "Audi", "Ford", "Chevrolet", "Other"], index=["Toyota", "Honda", "Tesla", "BMW", "Mercedes", "Audi", "Ford", "Chevrolet", "Other"].index(d_make) if d_make in ["Toyota", "Honda", "Tesla", "BMW", "Mercedes", "Audi", "Ford", "Chevrolet", "Other"] else 0)
            auto_year = st.slider("Model Year", min_value=1998, max_value=2026, value=d_year)
            property_damage = st.selectbox("Third-Party Property Damage", ["NO", "YES", "Unknown"])

        with col_c:
            st.markdown("#### 💥 3. Incident & Verification")
            type_opts = ["Single Vehicle Collision", "Multi-vehicle Collision", "Vehicle Theft", "Parked Car"]
            incident_type = st.selectbox("Incident Type", type_opts, index=type_opts.index(d_type) if d_type in type_opts else 0)
            
            sev_opts = ["Minor Damage", "Major Damage", "Total Loss", "Trivial Damage"]
            incident_severity = st.selectbox("Incident Severity", sev_opts, index=sev_opts.index(d_sev) if d_sev in sev_opts else 1)
            
            col_opts = ["Front Collision", "Rear Collision", "Side Collision", "Unknown"]
            collision_type = st.selectbox("Collision Point", col_opts, index=col_opts.index(d_col) if d_col in col_opts else 0)
            
            auth_opts = ["Police", "Fire", "Ambulance", "Other", "Unknown", "None"]
            authorities_contacted = st.selectbox("Authorities Notified", auth_opts, index=auth_opts.index(d_auth) if d_auth in auth_opts else 0)
            
            witnesses = st.slider("Witness Count", min_value=0, max_value=8, value=d_wit)
            police_report_available = st.selectbox("Police Report Available", ["YES", "NO", "Unknown"], index=["YES", "NO", "Unknown"].index(d_rep))
            bodily_injuries = st.slider("Bodily Injuries Incurred", min_value=0, max_value=4, value=d_inj)

        eval_submitted = st.form_submit_button("⚡ Execute AI Fraud Risk & Routing Assessment", use_container_width=True)

    # Evaluation Execution & Display
    claim_payload = {
        "customer_age": age,
        "months_as_customer": months_customer,
        "previous_claims": previous_claims,
        "insured_sex": insured_sex,
        "insured_education_level": education_level,
        "insured_occupation": occupation,
        "policy_state": policy_state,
        "claim_amount": claim_amount,
        "policy_annual_premium": policy_annual_premium,
        "policy_deductable": policy_deductable,
        "auto_make": auto_make,
        "auto_year": auto_year,
        "property_damage": property_damage,
        "incident_type": incident_type,
        "incident_severity": incident_severity,
        "collision_type": collision_type,
        "authorities_contacted": authorities_contacted,
        "witnesses": witnesses,
        "police_report_available": police_report_available,
        "bodily_injuries": bodily_injuries
    }

    results = compute_claim_assessment(claim_payload)

    st.markdown("---")
    st.markdown("### 📊 Official Claim Risk & Routing Audit")

    # Top Executive Summary Row
    res_col1, res_col2, res_col3 = st.columns([1.2, 1.4, 1.4])
    
    with res_col1:
        st.markdown(f"""
        <div class="metric-card">
            <span class="status-badge {results['badge_style']}">{results['risk_category']}</span>
            <div style="margin-top: 0.75rem;">
                <span class="{ 'risk-score-high' if results['composite_prob'] > 0.7 else ('risk-score-mid' if results['composite_prob'] > 0.3 else 'risk-score-low') }">
                    {results['composite_prob']:.1%}
                </span>
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 500;">Calibrated Risk Index</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(results['composite_prob']))

    with res_col2:
        st.markdown(f"""
        <div class="metric-card" style="text-align: left;">
            <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Routing Decision</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin: 0.25rem 0;">
                { '🟢' if results['composite_prob'] <= 0.3 else ('🟡' if results['composite_prob'] <= 0.7 else '🔴') } {results['routing_title']}
            </div>
            <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.4;">
                {results['action_summary']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with res_col3:
        st.markdown(f"""
        <div class="metric-card" style="text-align: left;">
            <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Operational Status</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #38bdf8; margin: 0.25rem 0;">
                {results['escrow_status']}
            </div>
            <div style="font-size: 0.85rem; color: #cbd5e1;">
                • Claim Exposure: <b>${claim_amount:,.2f}</b><br>
                • Deductible: <b>${policy_deductable:,.2f}</b><br>
                • Police Report: <b>{police_report_available}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Detailed Dual-Engine Breakdown & Factor Audit
    st.markdown("<br>", unsafe_allow_html=True)
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        st.markdown("#### ⚙️ Dual-Engine Inference Telemetry")
        st.markdown(f"""
        | Evaluation Engine | Probability Output | Inference Latency | Target Function |
        | :--- | :---: | :---: | :--- |
        | **XGBoost Teacher** | `{results['teacher_prob']:.2%}` | `{results['teacher_time']:.2f} ms` | Tree Boosting (`scale_pos_weight=19.1`) |
        | **PyTorch Student** | `{results['student_prob']:.2%}` | `{results['student_time']:.2f} ms` | Distilled Neural Net (`4-Layer FC`) |
        | **Consensus Score** | **`{results['composite_prob']:.2%}`** | `< 5.0 ms` | Tiered Decision Protocol |
        """)

    with d_col2:
        st.markdown("#### 🔍 Identified Risk Indicators")
        for title, desc in results['risk_factors']:
            st.markdown(f"""
            <div class="factor-item">
                <b>{title}</b><br>
                <span style="color: #cbd5e1; font-size: 0.825rem;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 2: MODEL BENCHMARK (TABLE 1)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("📊 Experimental Results from Paper (Table 1)")
    st.markdown("""
    Evaluation on the **20,000-record test set** (80:20 stratified split of 100,000 insurance records) 
    demonstrates the trade-offs between gradient-boosted decision trees and compact neural distillation.
    """)
    
    benchmark_data = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        "Teacher (XGBoost)": ["83.98%", "5.30%", "13.17%", "7.56%", "0.5018"],
        "Student (PyTorch Distilled)": ["95.03%", "0.00%", "0.00%", "0.00%", "0.4963"],
        "Operational Insight": [
            "Baseline accuracy reflecting positive class weighting (19.0954 : 1)",
            "Captures minority fraud patterns with 13.17% recall",
            "High student accuracy caused by majority class dominance",
            "Zero precision/recall due to standard 0.5 decision threshold",
            "Indicates the requirement for probability-based routing thresholds"
        ]
    })
    st.dataframe(benchmark_data, use_container_width=True, hide_index=True)
    
    st.divider()
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("#### ⚖️ Dataset & Imbalance Profile")
        st.markdown("""
        - **Total Records:** 100,000 claims (80,000 train / 20,000 test)
        - **Class 0 (Genuine):** 76,019 samples (95.02%)
        - **Class 1 (Historical Fraud):** 3,981 samples (4.98%)
        - **Positive Class Weight:** `19.0954`
        - **Preprocessed Feature Dimension:** 93 continuous/one-hot attributes
        """)
    with b2:
        st.markdown("#### 📉 Distillation Training Curve")
        st.markdown("""
        - **Initial Distillation Loss (Epoch 1):** `5.7274`
        - **Final Distillation Loss (Epoch 15):** `5.7116`
        - **Distillation Temperature ($T$):** `4.0`
        - **Loss Balancing Weight ($\alpha$):** `0.5`
        """)

# ------------------------------------------------------------------------------
# TAB 3: BATCH & RANDOM STRESS TESTING
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("🎲 Batch Simulation & Random Stress Testing")
    st.markdown("Simulate multiple claims with randomized parameters to stress-test the model's routing engine.")
    
    num_sim = st.slider("Select Number of Synthetic Claims to Simulate", min_value=1, max_value=10, value=3)
    
    if st.button("🚀 Run Batch Claim Simulation", type="primary"):
        sim_records = []
        for i in range(num_sim):
            r_age = random.randint(19, 80)
            r_amt = round(random.uniform(2000.0, 140000.0), 2)
            r_make = random.choice(["Toyota", "Honda", "Tesla", "BMW", "Mercedes", "Audi", "Ford"])
            r_year = random.randint(2002, 2025)
            r_type = random.choice(["Single Vehicle Collision", "Multi-vehicle Collision", "Vehicle Theft", "Parked Car"])
            r_sev = random.choice(["Minor Damage", "Major Damage", "Total Loss", "Trivial Damage"])
            r_rep = random.choice(["YES", "NO", "Unknown"])
            r_wit = random.randint(0, 4)
            r_inj = random.randint(0, 3)
            r_claims = random.randint(0, 4)

            payload = {
                "customer_age": r_age,
                "claim_amount": r_amt,
                "auto_make": r_make,
                "auto_year": r_year,
                "incident_type": r_type,
                "incident_severity": r_sev,
                "police_report_available": r_rep,
                "witnesses": r_wit,
                "bodily_injuries": r_inj,
                "previous_claims": r_claims
            }
            res = compute_claim_assessment(payload)
            sim_records.append({
                "Claim ID": f"CLM-{random.randint(10000, 99999)}",
                "Customer Age": r_age,
                "Vehicle": f"{r_make} ({r_year})",
                "Claim Amount": f"${r_amt:,.2f}",
                "Incident": f"{r_type} ({r_sev})",
                "Police Report": r_rep,
                "Teacher Prob": f"{res['teacher_prob']:.2%}",
                "Student Prob": f"{res['student_prob']:.2%}",
                "Risk Score": f"{res['composite_prob']:.2%}",
                "Routing Decision": res['routing_title']
            })
            
        st.dataframe(pd.DataFrame(sim_records), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 4: KNOWLEDGE DISTILLATION ARCHITECTURE
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("🔬 Knowledge Distillation Architecture & Loss Formulation")
    st.markdown("""
    ### 1. The Distillation Loss Function
    The student network is trained using a composite loss function that combines hard ground-truth supervision with soft probability matching from the teacher:
    
    $$\mathcal{L} = \\alpha \mathcal{L}_{hard}(y, \hat{y}_{student}) + (1 - \\alpha) T^2 \mathcal{L}_{soft}(q_{teacher}, q_{student})$$
    
    where:
    - $\mathcal{L}_{hard}$ is **Binary Cross-Entropy Loss** with respect to ground-truth labels.
    - $\mathcal{L}_{soft}$ is **Softened Binary Cross-Entropy Loss** matching temperature-softened student logits against teacher logits.
    - $T = 4.0$ is the **Distillation Temperature**, smoothing probability distributions to expose dark knowledge.
    - $\\alpha = 0.5$ balances the gradient updates between ground truth and teacher supervision.
    - $T^2$ scales the gradient magnitudes so that soft loss gradients match the scale of hard loss gradients.

    ### 2. Network Specifications
    ```
    Input Layer   : 93 Features (Median Imputation + Standard Scaler / Most Frequent + One-Hot)
    Hidden Layer 1: Linear(93 -> 128) + ReLU
    Hidden Layer 2: Linear(128 -> 64) + ReLU
    Hidden Layer 3: Linear(64 -> 32)  + ReLU
    Output Layer  : Linear(32 -> 1)   + Sigmoid (Inference)
    Optimizer     : Adam (lr = 0.001)
    Epochs        : 15
    ```
    """)
