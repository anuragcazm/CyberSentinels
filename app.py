import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="CyberPredict Intelligence API // PS-26184",
    version="1.0.0",
    description="Predictive Analytics Framework for Cybercrime Complaints to Forecast Cash-Out Locations"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../ml_engine/cyberpredict_rf_model.pkl")
if os.path.exists(MODEL_PATH):
    ml_model = joblib.load(MODEL_PATH)
else:
    ml_model = None

REGIONAL_MATRIX = {
    "east": {
        "name": "Laxmi Nagar Metro ATM & CSP Hub",
        "subtext": "East Delhi • Vikas Marg / Shakarpur",
        "lat": 28.6304,
        "lng": 77.2773,
        "hist_density": 88.0,
        "station": "East Delhi Cyber Police Station",
        "officer": "Inspector Vikramaditya Sharma (SHO)",
        "phone": "+91 11 2244 5566 / 98711-CYBER",
        "i4c_node": "Delhi State Cyber Coordination Unit (I4C-DL-02)",
        "bank_nodal": "SBI Vikas Marg & HDFC Laxmi Nagar Desk",
        "bank_contact": "+91 98112-BANKN"
    },
    "central": {
        "name": "Connaught Place Financial Circle ATMs",
        "subtext": "Central Delhi • Scindia House Nodes",
        "lat": 28.6315,
        "lng": 77.2167,
        "hist_density": 82.0,
        "station": "New Delhi District Cyber Cell",
        "officer": "ACP Rajesh Verma / Insp. Anita Roy",
        "phone": "+91 11 2374 8899 / 99100-CYBER",
        "i4c_node": "Central Delhi I4C Coordination Desk",
        "bank_nodal": "PNB Sansad Marg & Axis Bank Hub",
        "bank_contact": "+91 98730-PNBNL"
    },
    "noida": {
        "name": "Noida Sector 18 Commercial Bank Center",
        "subtext": "Gautam Buddha Nagar • Atta Market",
        "lat": 28.5708,
        "lng": 77.3260,
        "hist_density": 78.0,
        "station": "Noida Cyber Police Station (Sector 108)",
        "officer": "DSP Sunil Kumar (Cyber Cell Noida)",
        "phone": "+91 120 256 7890 / 94544-CYBER",
        "i4c_node": "UP State Cyber Crime Unit (I4C-UP-08)",
        "bank_nodal": "ICICI Sec-18 & Bank of Baroda Hub",
        "bank_contact": "+91 98991-ICICIN"
    },
    "northwest": {
        "name": "Rohini Sector 10 District Centre",
        "subtext": "North West Delhi • Swarn Jayanti Perimeter",
        "lat": 28.7159,
        "lng": 77.1147,
        "hist_density": 75.0,
        "station": "North-West District Cyber Cell, Rohini",
        "officer": "Inspector Devendra Hooda",
        "phone": "+91 11 2705 4433 / 98101-NWDP",
        "i4c_node": "Rohini Sub-Divisional Cyber Unit",
        "bank_nodal": "Canara Bank & Union Bank Rohini",
        "bank_contact": "+91 98188-NODAL"
    },
    "west": {
        "name": "Janakpuri District Commercial Complex",
        "subtext": "West Delhi • Major Commercial ATM Hub",
        "lat": 28.6219,
        "lng": 77.0878,
        "hist_density": 72.0,
        "station": "West District Cyber Police Station",
        "officer": "Inspector Kuldeep Rawat",
        "phone": "+91 11 2550 3322 / 98733-WESTCYBER",
        "i4c_node": "West Delhi District Cyber Operations",
        "bank_nodal": "HDFC District Center Janakpuri",
        "bank_contact": "+91 98115-HDFCN"
    },
    "gurugram": {
        "name": "Gurugram Cyber City Commercial Outlets",
        "subtext": "Haryana • DLF Phase 2 Corridor",
        "lat": 28.4952,
        "lng": 77.0895,
        "hist_density": 68.0,
        "station": "Cyber Crime Police Station East Gurugram",
        "officer": "ACP Priyanshu Dewan / Insp. Jasvir Singh",
        "phone": "+91 124 221 1100 / 99999-HRYCYBER",
        "i4c_node": "Haryana State Cyber Crime Coordination Center",
        "bank_nodal": "Kotak Mahindra & IndusInd Cyber City",
        "bank_contact": "+91 98109-CYBERBANK"
    }
}

active_hotspots_db = [
    {
        "id": "NCRP-2026-901",
        "name": REGIONAL_MATRIX["east"]["name"],
        "subtext": REGIONAL_MATRIX["east"]["subtext"],
        "lat": REGIONAL_MATRIX["east"]["lat"],
        "lng": REGIONAL_MATRIX["east"]["lng"],
        "score": 93.4,
        "level": "CRITICAL",
        "amount": "₹4,80,000",
        "lead_time": "12 min",
        "category": "UPI / QR Code Scam",
        "past_30d_incidents": 48,
        "past_90d_incidents": 134,
        "assigned_station": REGIONAL_MATRIX["east"]["station"],
        "assigned_officer": REGIONAL_MATRIX["east"]["officer"],
        "emergency_phone": REGIONAL_MATRIX["east"]["phone"],
        "bank_nodal": REGIONAL_MATRIX["east"]["bank_nodal"],
        "bank_phone": REGIONAL_MATRIX["east"]["bank_contact"],
        "shap_factors": ["Velocity Spike in Shakarpur", "Frequent Mule Off-Ramp", "Off-hours Withdrawal Pattern"]
    },
    {
        "id": "NCRP-2026-902",
        "name": REGIONAL_MATRIX["central"]["name"],
        "subtext": REGIONAL_MATRIX["central"]["subtext"],
        "lat": REGIONAL_MATRIX["central"]["lat"],
        "lng": REGIONAL_MATRIX["central"]["lng"],
        "score": 89.2,
        "level": "CRITICAL",
        "amount": "₹6,20,000",
        "lead_time": "18 min",
        "category": "Part-Time Task Scam",
        "past_30d_incidents": 39,
        "past_90d_incidents": 112,
        "assigned_station": REGIONAL_MATRIX["central"]["station"],
        "assigned_officer": REGIONAL_MATRIX["central"]["officer"],
        "emergency_phone": REGIONAL_MATRIX["central"]["phone"],
        "bank_nodal": REGIONAL_MATRIX["central"]["bank_nodal"],
        "bank_phone": REGIONAL_MATRIX["central"]["bank_contact"],
        "shap_factors": ["Syndicate Account Mule Chain", "High-value cardless transaction alert"]
    },
    {
        "id": "NCRP-2026-903",
        "name": REGIONAL_MATRIX["noida"]["name"],
        "subtext": REGIONAL_MATRIX["noida"]["subtext"],
        "lat": REGIONAL_MATRIX["noida"]["lat"],
        "lng": REGIONAL_MATRIX["noida"]["lng"],
        "score": 81.5,
        "level": "HIGH",
        "amount": "₹3,40,000",
        "lead_time": "32 min",
        "category": "Investment / Ponzi Fraud",
        "past_30d_incidents": 31,
        "past_90d_incidents": 89,
        "assigned_station": REGIONAL_MATRIX["noida"]["station"],
        "assigned_officer": REGIONAL_MATRIX["noida"]["officer"],
        "emergency_phone": REGIONAL_MATRIX["noida"]["phone"],
        "bank_nodal": REGIONAL_MATRIX["noida"]["bank_nodal"],
        "bank_phone": REGIONAL_MATRIX["noida"]["bank_contact"],
        "shap_factors": ["High fraud value concentration", "Rapid AePS disbursement link"]
    }
]

class FraudComplaintRequest(BaseModel):
    victim_name: str
    category: str
    amount_inr: float
    region_key: str
    mule_hops: int
    mule_account: str
    hour_of_day: int = 14

@app.get("/")
def root():
    return {"status": "ONLINE", "service": "CyberPredict AI Platform", "problem_statement": "26184"}

@app.get("/api/v1/hotspots")
def get_all_hotspots():
    sorted_threats = sorted(active_hotspots_db, key=lambda x: x["score"], reverse=True)
    return {"status": "success", "count": len(sorted_threats), "data": sorted_threats}

@app.post("/api/v1/report-fraud")
def report_fraud_and_auto_assign(payload: FraudComplaintRequest):
    region = REGIONAL_MATRIX.get(payload.region_key)
    if not region:
        raise HTTPException(status_code=400, detail="Invalid region specified.")

    features = np.array([[
        payload.amount_inr,
        payload.hour_of_day,
        payload.mule_hops,
        region["hist_density"]
    ]])

    if ml_model is not None:
        raw_score = float(ml_model.predict(features)[0])
    else:
        raw_score = 0.4 * (payload.amount_inr / 100000 * 100) + 0.4 * region["hist_density"] + 0.2 * (100 - payload.mule_hops * 20)

    risk_score = round(float(np.clip(raw_score, 20.0, 97.5)), 1)
    lead_time_min = max(8, int(55 - (risk_score * 0.42)))
    complaint_id = f"NCRP-2026-{np.random.randint(1000, 9999)}"

    jitter_lat = region["lat"] + float(np.random.normal(0, 0.003))
    jitter_lng = region["lng"] + float(np.random.normal(0, 0.003))

    new_incident = {
        "id": complaint_id,
        "name": f"{region['name']} (LIVE ALERT)",
        "subtext": f"Victim: {payload.victim_name} • Account: {payload.mule_account}",
        "lat": round(jitter_lat, 5),
        "lng": round(jitter_lng, 5),
        "score": risk_score,
        "level": "CRITICAL" if risk_score >= 85 else "HIGH",
        "amount": f"₹{int(payload.amount_inr):,}",
        "lead_time": f"{lead_time_min} min",
        "category": payload.category,
        "past_30d_incidents": np.random.randint(25, 60),
        "past_90d_incidents": np.random.randint(70, 150),
        "assigned_station": region["station"],
        "assigned_officer": region["officer"],
        "emergency_phone": region["phone"],
        "bank_nodal": region["bank_nodal"],
        "bank_phone": region["bank_contact"],
        "shap_factors": [
            "Incoming 1930 live telemetry",
            f"Mule layer depth: {payload.mule_hops} hop(s)",
            "Automated jurisdictional assignment"
        ]
    }

    active_hotspots_db.insert(0, new_incident)

    return {
        "status": "success",
        "message": "Complaint ingested and directly assigned to Cyber Police Station.",
        "assigned_case": new_incident
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
