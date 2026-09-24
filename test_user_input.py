import sys
from predict_claim import load_inference_artifacts, predict_single_claim

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 1. Load trained models & preprocessor
preprocessor, template_df, teacher_model, student_model = load_inference_artifacts()

# 2. Example User Input Data
user_inputs = {
    "customer_age": 42,
    "claim_amount": 85000.00,
    "incident_severity": "Total Loss",
    "incident_type": "Multi-vehicle Collision",
    "collision_type": "Front Collision",
    "authorities_contacted": "Police",
    "witnesses": 2,
    "police_report_available": "YES",
    "bodily_injuries": 1
}

# 3. Predict with Dual-Engine (XGBoost + Distilled PyTorch)
prob, decision, t_prob, s_prob = predict_single_claim(
    preprocessor, template_df, teacher_model, student_model, user_inputs
)

# 4. Display Results
print("=" * 68)
print("         ENTERPRISE CLAIM FRAUD RISK AUDIT DEMO")
print("=" * 68)
print(f" Customer Age        : {user_inputs['customer_age']} years")
print(f" Total Claim Amount  : ${user_inputs['claim_amount']:,.2f}")
print(f" Incident Type       : {user_inputs['incident_type']}")
print(f" Incident Severity   : {user_inputs['incident_severity']}")
print(f" Collision Point     : {user_inputs['collision_type']}")
print(f" Authorities Notified: {user_inputs['authorities_contacted']}")
print(f" Witnesses Present   : {user_inputs['witnesses']}")
print(f" Police Report       : {user_inputs['police_report_available']}")
print("-" * 68)
print(f" [AI TELEMETRY] XGBoost Teacher Prob : {t_prob:.2%}")
print(f" [AI TELEMETRY] PyTorch Student Prob : {s_prob:.2%}")
print(f" [RESULT]       COMPOSITE FRAUD RISK : {prob:.2%}")
print(f" [DECISION]     AUTOMATED ROUTING    : {decision}")
print("=" * 68)
