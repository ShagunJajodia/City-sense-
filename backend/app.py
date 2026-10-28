from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
import random
import os
import asyncio

app = FastAPI(title="CitySense API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "../ml/models")

try:
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    encoders = joblib.load(os.path.join(model_dir, "encoders.pkl"))
    explainer = joblib.load(os.path.join(model_dir, "explainer.pkl"))
    with open(os.path.join(model_dir, "features.json"), "r") as f:
        features_list = json.load(f)
    print("CitySense Models & Explainer loaded successfully.")
except Exception as e:
    print(f"Warning: Models error. {e}")

BENGALURU_ZONES = [
    {"name": "Whitefield", "lat": 12.9698, "lon": 77.7499, "base_aqi": 180, "base_water": "Fair", "base_traffic": "Severe", "transport": 8},
    {"name": "Electronic City", "lat": 12.8399, "lon": 77.6770, "base_aqi": 140, "base_water": "Good", "base_traffic": "High", "transport": 7},
    {"name": "Bellandur", "lat": 12.9304, "lon": 77.6784, "base_aqi": 190, "base_water": "Poor", "base_traffic": "Severe", "transport": 6},
    {"name": "Marathahalli", "lat": 12.9569, "lon": 77.7011, "base_aqi": 160, "base_water": "Fair", "base_traffic": "High", "transport": 7},
    {"name": "Indiranagar", "lat": 12.9784, "lon": 77.6408, "base_aqi": 90, "base_water": "Good", "base_traffic": "High", "transport": 8},
    {"name": "Koramangala", "lat": 12.9279, "lon": 77.6271, "base_aqi": 100, "base_water": "Good", "base_traffic": "High", "transport": 9},
    {"name": "Peenya", "lat": 13.0285, "lon": 77.5197, "base_aqi": 300, "base_water": "Poor", "base_traffic": "High", "transport": 5},
    {"name": "Majestic", "lat": 12.9766, "lon": 77.5713, "base_aqi": 160, "base_water": "Fair", "base_traffic": "Severe", "transport": 10},
    {"name": "KR Market", "lat": 12.9651, "lon": 77.5752, "base_aqi": 180, "base_water": "Poor", "base_traffic": "Severe", "transport": 9},
    {"name": "Malleswaram", "lat": 13.0031, "lon": 77.5643, "base_aqi": 70, "base_water": "Good", "base_traffic": "Normal", "transport": 8},
    {"name": "Hebbal", "lat": 13.0354, "lon": 77.5988, "base_aqi": 140, "base_water": "Fair", "base_traffic": "High", "transport": 7},
    {"name": "Silk Board", "lat": 12.9176, "lon": 77.6234, "base_aqi": 150, "base_water": "Fair", "base_traffic": "Severe", "transport": 8},
    {"name": "Jayanagar", "lat": 12.9299, "lon": 77.5826, "base_aqi": 60, "base_water": "Good", "base_traffic": "Normal", "transport": 8},
    {"name": "HSR Layout", "lat": 12.9121, "lon": 77.6446, "base_aqi": 85, "base_water": "Good", "base_traffic": "High", "transport": 8},
    {"name": "BTM Layout", "lat": 12.9166, "lon": 77.6101, "base_aqi": 95, "base_water": "Fair", "base_traffic": "High", "transport": 8},
    {"name": "Yeshwanthpur", "lat": 13.0224, "lon": 77.5555, "base_aqi": 170, "base_water": "Poor", "base_traffic": "Severe", "transport": 10},
    {"name": "Rajajinagar", "lat": 12.9982, "lon": 77.5530, "base_aqi": 80, "base_water": "Good", "base_traffic": "Normal", "transport": 8},
    {"name": "Banashankari", "lat": 12.9255, "lon": 77.5468, "base_aqi": 75, "base_water": "Good", "base_traffic": "Normal", "transport": 7},
    {"name": "Ulsoor", "lat": 12.9818, "lon": 77.6284, "base_aqi": 110, "base_water": "Fair", "base_traffic": "High", "transport": 9},
    {"name": "Frazer Town", "lat": 12.9988, "lon": 77.6166, "base_aqi": 90, "base_water": "Good", "base_traffic": "Normal", "transport": 8},
    {"name": "Yelahanka", "lat": 13.1007, "lon": 77.5963, "base_aqi": 65, "base_water": "Good", "base_traffic": "Low", "transport": 6},
    {"name": "Vijayanagar", "lat": 12.9756, "lon": 77.5354, "base_aqi": 70, "base_water": "Good", "base_traffic": "Normal", "transport": 7},
    {"name": "CV Raman Nagar", "lat": 12.9790, "lon": 77.6644, "base_aqi": 100, "base_water": "Fair", "base_traffic": "High", "transport": 6},
    {"name": "RT Nagar", "lat": 13.0247, "lon": 77.5948, "base_aqi": 120, "base_water": "Fair", "base_traffic": "High", "transport": 7},
    {"name": "Richmond Town", "lat": 12.9647, "lon": 77.6015, "base_aqi": 85, "base_water": "Good", "base_traffic": "Normal", "transport": 8},
    {"name": "Nelamangala", "lat": 13.0970, "lon": 77.3915, "base_aqi": 80, "base_water": "Fair", "base_traffic": "High", "transport": 6},
    {"name": "Doddaballapura", "lat": 13.2924, "lon": 77.5434, "base_aqi": 70, "base_water": "Good", "base_traffic": "Normal", "transport": 5},
    {"name": "Hoskote", "lat": 13.0722, "lon": 77.7975, "base_aqi": 120, "base_water": "Poor", "base_traffic": "High", "transport": 6},
    {"name": "Devanahalli", "lat": 13.2430, "lon": 77.7140, "base_aqi": 90, "base_water": "Good", "base_traffic": "High", "transport": 9},
    {"name": "Bidadi", "lat": 12.7984, "lon": 77.3879, "base_aqi": 100, "base_water": "Fair", "base_traffic": "Normal", "transport": 6},
    {"name": "Kengeri", "lat": 12.8996, "lon": 77.4827, "base_aqi": 75, "base_water": "Good", "base_traffic": "High", "transport": 8},
    {"name": "Anekal", "lat": 12.7099, "lon": 77.6999, "base_aqi": 85, "base_water": "Fair", "base_traffic": "Normal", "transport": 5},
    {"name": "Ramanagara", "lat": 12.7150, "lon": 77.2812, "base_aqi": 60, "base_water": "Good", "base_traffic": "Normal", "transport": 5},
    {"name": "Magadi", "lat": 12.9566, "lon": 77.2346, "base_aqi": 50, "base_water": "Good", "base_traffic": "Low", "transport": 4},
    {"name": "Kanakapura", "lat": 12.5484, "lon": 77.4140, "base_aqi": 55, "base_water": "Good", "base_traffic": "Low", "transport": 4}
]

class PredictionRequest(BaseModel):
    Zone: str
    Traffic: str
    Weather: str
    Road_Condition: str
    Pollution_AQI: int
    Water_Quality: str
    Transport_Connection: int

@app.get("/metrics")
def get_metrics():
    try:
        with open(os.path.join(model_dir, "model_metrics.json"), "r") as f:
            metrics = json.load(f)
        return metrics
    except:
        return {"error": "Metrics not found"}

@app.get("/zones")
def get_zones():
    return {"zones": BENGALURU_ZONES}

@app.post("/predict")
def predict_risk(req: PredictionRequest):
    input_data = req.dict()
    df = pd.DataFrame([input_data])
    
    categorical_cols = ["Zone", "Traffic", "Weather", "Road_Condition", "Water_Quality"]
    for col in categorical_cols:
        if input_data[col] in encoders[col].classes_:
            df[col] = encoders[col].transform([input_data[col]])
        else:
            df[col] = encoders[col].transform([encoders[col].classes_[0]])
            
    df = df[features_list]

    pred_idx = int(model.predict(df)[0])
    pred_label = encoders["Accident_Prone_Level"].inverse_transform([pred_idx])[0]
    
    # SHAP Explainable AI Logic
    xai_reason = "Unable to compute feature importance."
    try:
        shap_vals = explainer.shap_values(df)
        if isinstance(shap_vals, list): 
            class_shap = shap_vals[pred_idx][0]
        else:
            # Handle ndarray of shape (n_samples, n_features, n_classes)
            if len(shap_vals.shape) == 3:
                class_shap = shap_vals[0, :, pred_idx]
            else:
                class_shap = shap_vals[0]
            
        max_idx = np.argmax(np.abs(class_shap))
        top_feature = features_list[max_idx]
        top_impact = class_shap[max_idx]
        
        impact_dir = "increased" if top_impact > 0 else "decreased"
        xai_reason = f"+{abs(top_impact)*100:.1f}% risk logic fluctuation due to {top_feature.replace('_', ' ')}."
    except Exception as e:
        print("SHAP Warning:", e)

    # Livability Index Calculation
    livability = 100
    if req.Pollution_AQI > 50: livability -= min(30, (req.Pollution_AQI-50)/5)
    if req.Traffic == "Severe": livability -= 20
    elif req.Traffic == "High": livability -= 10
    if req.Water_Quality == "Poor": livability -= 15
    elif req.Water_Quality == "Fair": livability -= 5
    livability += (req.Transport_Connection - 5) * 2
    livability = max(0, min(100, livability))
    
    # Generative AI Fake-LLM Advisory Logic
    advisory = ""
    if pred_label == "High":
        advisory = f"🚨 CitySense LLM Advisory: High accident risk detected in {req.Zone}. Immediate municipal action recommended. Current {req.Traffic.lower()} traffic coupled with localized infrastructure states requires preemptive dispatch of traffic management authorities."
    elif livability < 60:
        advisory = f"⚠️ CitySense LLM Advisory: Livability dropping in {req.Zone} primarily due to AQI levels of {req.Pollution_AQI} and {req.Water_Quality.lower()} water quality. Suggesting long-term infrastructure funding."
    else:
        advisory = f"✅ CitySense LLM Advisory: {req.Zone} is operating nominally. Traffic flow is {req.Traffic.lower()} with low immediate accident risks. No predictive maintenance required."

    return {
        "accident_level": pred_label,
        "livability_index": round(livability, 1),
        "xai_reason": xai_reason,
        "ai_advisory": advisory
    }

@app.websocket("/ws/live_simulate")
async def websocket_endpoint(websocket: WebSocket):
    """
    True Websocket implementation for streaming continuous IoT data
    """
    await websocket.accept()
    try:
        while True:
            # Stream 5 random zones every 2 seconds
            simulations = []
            selected = random.sample(BENGALURU_ZONES, 5)
            
            for zone in selected:
                req = PredictionRequest(
                    Zone=zone["name"],
                    Traffic=random.choice(["Low", "Normal", "High", "Severe"]),
                    Weather=random.choice(["Clear", "Rain", "Heavy Rain", "Fog"]),
                    Road_Condition=random.choice(["Excellent", "Good", "Average", "Poor", "Potholed"]),
                    Pollution_AQI=random.randint(50, 400),
                    Water_Quality=random.choice(["Good", "Fair", "Poor"]),
                    Transport_Connection=random.randint(2, 10)
                )
                res = predict_risk(req)
                simulations.append({
                    "zone_info": zone,
                    "inputs": req.dict(),
                    "prediction": res
                })
            
            await websocket.send_json({"simulations": simulations})
            await asyncio.sleep(2)
            
    except Exception as e:
        print("Websocket closed", e)

frontend_dir = os.path.join(base_dir, "../frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

