# CyberPredict // SIH Problem Statement 26184
### Development of a Predictive Analytics Framework for Cybercrime Complaints to Forecast Likely Cash Withdrawal Locations

---

## 🌟 Overview
CyberPredict is an AI/ML spatiotemporal framework designed for the Indian Cyber Crime Coordination Centre (I4C) and the National Cybercrime Reporting Portal (NCRP / 1930 Helpline). It shifts cybercrime response from reactive post-facto investigation to **proactive cash-out intervention**, predicting physical ATM/CSP liquidation locations before stolen funds are withdrawn.

---

## 📂 Project Architecture
```
cyberpredict_project/
├── backend/
│   ├── app.py                # FastAPI REST Server & Scoring Engine
│   └── requirements.txt      # Python Dependencies
├── ml_engine/
│   └── cyberpredict_rf_model.pkl  # Trained Random Forest Model
├── data/
│   └── cyberpredict_dataset.csv   # 500 records synthetic spatiotemporal dataset
├── frontend/
│   └── index.html            # Tactical GIS Command Dashboard (Leaflet + Tailwind)
└── README.md                 # Setup & Run Guide
```

---

## 🚀 Quickstart & Execution Steps

### Step 1: Install Python Dependencies
```bash
cd cyberpredict_project/backend
pip install -r requirements.txt
```

### Step 2: Start the FastAPI Backend Server
```bash
python app.py
```
* Server will start at: `http://127.0.0.1:8000`
* Interactive API Documentation (Swagger): `http://127.0.0.1:8000/docs`

### Step 3: Launch the Tactical Command Dashboard
Open `cyberpredict_project/frontend/index.html` in any web browser (Chrome, Edge, Firefox, Safari).

---

## 🛠 Features Included
1. **Interactive GIS Map**: Real-time Leaflet hotspot visualization.
2. **Instant Fraud Reporting Modal**: Log incoming 1930 complaints with victim metadata.
3. **ML Risk Scoring**: Real-time Random Forest inference (0–100 score + estimated lead time).
4. **Auto-Assignment to Police**: Maps complaints to the exact jurisdictional Cyber Police Station, SHO/ACP contacts, and I4C nodes.
5. **Dual Dispatch System**: One-click notification triggers for Police Patrols and Bank Nodal Officers.
