import sys
import random
from predict_claim import load_inference_artifacts, predict_single_claim

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

preprocessor, template_df, teacher_model, student_model = load_inference_artifacts()

def generate_random_claim():
    return {
        "customer_age": random.randint(18, 85),
        "claim_amount": round(random.uniform(1500.0, 135000.0), 2),
        "months_as_customer": random.randint(1, 300),
        "previous_claims": random.randint(0, 5),
        "insured_sex": random.choice(["MALE", "FEMALE"]),
        "insured_education_level": random.choice(["Associate", "PhD", "High School", "Bachelor", "MD"]),
        "insured_occupation": random.choice(["Teacher", "Doctor", "Engineer", "Technician", "Lawyer", "Sales"]),
        "policy_state": random.choice(["TX", "CA", "OH", "NY", "FL", "IL"]),
        "incident_type": random.choice(["Single Vehicle Collision", "Vehicle Theft", "Multi-vehicle Collision", "Parked Car"]),
        "incident_severity": random.choice(["Minor Damage", "Total Loss", "Major Damage", "Trivial Damage"]),
        "collision_type": random.choice(["Front Collision", "Rear Collision", "Side Collision", "Unknown"]),
        "authorities_contacted": random.choice(["Police", "Fire", "Ambulance", "None", "Unknown"]),
        "witnesses": random.randint(0, 5),
        "police_report_available": random.choice(["YES", "NO", "Unknown"]),
        "bodily_injuries": random.randint(0, 4),
        "auto_make": random.choice(["Toyota", "Tesla", "Honda", "BMW", "Ford", "Audi", "Mercedes"]),
        "auto_year": random.randint(2000, 2025)
    }

print("=" * 70)
print("          TESTING 3 RANDOM SYNTHETIC CLAIMS (DUAL-ENGINE)")
print("=" * 70)

for i in range(1, 4):
    r_claim = generate_random_claim()
    prob, decision, t_prob, s_prob = predict_single_claim(
        preprocessor, template_df, teacher_model, student_model, r_claim
    )
    
    print(f"\n[CLAIM #{i}]")
    print(f" Profile: {r_claim['customer_age']} yrs | Vehicle: {r_claim['auto_make']} ({r_claim['auto_year']}) | Amount: ${r_claim['claim_amount']:,.2f}")
    print(f" Incident: {r_claim['incident_type']} ({r_claim['incident_severity']}) | Witnesses: {r_claim['witnesses']} | Police Report: {r_claim['police_report_available']}")
    print(f" Engine Telemetry -> XGBoost: {t_prob:.2%} | Distilled PyTorch: {s_prob:.2%}")
    print(f" Final Decision  -> Risk Index: {prob:.2%} | Routing: {decision}")
    print("-" * 70)

print("\n" + "=" * 70)
print("BENCHMARK METRICS (Table 1 in Paper):")
print("Teacher Accuracy : 83.98% | Recall: 13.17% | ROC-AUC: 0.5018")
print("Student Accuracy : 95.03% | Imbalance: 19.0954 : 1")
print("=" * 70)
