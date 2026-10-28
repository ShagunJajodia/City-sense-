import pandas as pd
import numpy as np
import random

# Base configurations for Bengaluru Zones
ZONES = [
    {"name": "Whitefield", "lat": 12.9698, "lon": 77.7499, "type": "IT_Hub"},
    {"name": "Electronic City", "lat": 12.8399, "lon": 77.6770, "type": "IT_Hub"},
    {"name": "Bellandur", "lat": 12.9304, "lon": 77.6784, "type": "IT_Hub+Lake"},
    {"name": "Marathahalli", "lat": 12.9569, "lon": 77.7011, "type": "Commercial"},
    {"name": "Indiranagar", "lat": 12.9784, "lon": 77.6408, "type": "Commercial_Nightlife"},
    {"name": "Koramangala", "lat": 12.9279, "lon": 77.6271, "type": "Commercial_Nightlife"},
    {"name": "Peenya", "lat": 13.0285, "lon": 77.5197, "type": "Industrial"},
    {"name": "Majestic", "lat": 12.9766, "lon": 77.5713, "type": "Transit_Hub"},
    {"name": "KR Market", "lat": 12.9651, "lon": 77.5752, "type": "Market"},
    {"name": "Malleswaram", "lat": 13.0031, "lon": 77.5643, "type": "Residential"},
    {"name": "Hebbal", "lat": 13.0354, "lon": 77.5988, "type": "Transit_Hub"},
    {"name": "Silk Board", "lat": 12.9176, "lon": 77.6234, "type": "Transit_Junction"},
    {"name": "Jayanagar", "lat": 12.9299, "lon": 77.5826, "type": "Residential"},
    {"name": "HSR Layout", "lat": 12.9121, "lon": 77.6446, "type": "Residential_Commercial"}
]

data = []
NUM_ROWS = 4000

for _ in range(NUM_ROWS):
    zone = random.choice(ZONES)
    
    # 1. Weather
    weather = random.choices(["Clear", "Rain", "Heavy Rain", "Fog"], weights=[60, 20, 10, 10])[0]
    
    # 2. Traffic
    # Base traffic depends on zone type
    traffic_opts = ["Low", "Normal", "High", "Severe"]
    if "Transit" in zone["type"] or "IT_Hub" in zone["type"]:
        base_traffic = random.choices(["Normal", "High", "Severe"], weights=[10, 50, 40])[0]
    elif "Residential" in zone["type"]:
        base_traffic = random.choices(["Low", "Normal", "High"], weights=[40, 40, 20])[0]
    else:
        base_traffic = random.choice(traffic_opts)
        
    # Weather amplifies traffic
    traffic = base_traffic
    if weather in ["Heavy Rain", "Fog"] and traffic != "Severe":
        curr_idx = traffic_opts.index(traffic)
        traffic = traffic_opts[min(3, curr_idx + 1)]
        
    # 3. Road Condition
    # Heavily trafficked or specific zones might have worse roads
    road_opts = ["Excellent", "Good", "Average", "Poor", "Potholed"]
    if "Transit" in zone["type"] or "Industrial" in zone["type"] or weather == "Heavy Rain":
        road_condition = random.choices(["Average", "Poor", "Potholed"], weights=[40, 40, 20])[0]
    else:
        road_condition = random.choices(["Excellent", "Good", "Average"], weights=[30, 50, 20])[0]
        
    # 4. Pollution AQI (50 to 500)
    base_aqi = 100
    if "Industrial" in zone["type"]: base_aqi = 250
    if traffic in ["High", "Severe"]: base_aqi += random.randint(50, 100)
    if weather == "Rain": base_aqi -= 30 # rain settles dust
    aqi = max(50, min(500, base_aqi + random.randint(-20, 40)))
    
    # 5. Water Quality (TDS / Contamination level: Good, Fair, Poor)
    water_quality = "Fair"
    if "Lake" in zone["type"]: water_quality = random.choices(["Fair", "Poor"], weights=[30, 70])[0]
    elif "Residential" in zone["type"]: water_quality = random.choices(["Good", "Fair"], weights=[70, 30])[0]
    else: water_quality = random.choice(["Good", "Fair", "Poor"])
    
    # 6. Transport Connection (1 to 10 scale of accessibility)
    if "Transit" in zone["type"] or "IT_Hub" in zone["type"]:
        transport_connection = random.randint(7, 10)
    elif "Industrial" in zone["type"]:
        transport_connection = random.randint(3, 6)
    else:
        transport_connection = random.randint(4, 8)
        
    # --- CALCULATE TARGET: Accident_Prone_Level ---
    # We invent a logical score then categorize it
    acc_score = 0
    
    # Traffic contributes significantly
    if traffic == "High": acc_score += 20
    elif traffic == "Severe": acc_score += 40
    
    # Weather contributes
    if weather == "Rain": acc_score += 15
    elif weather in ["Heavy Rain", "Fog"]: acc_score += 30
    
    # Road Condition contributes heavily
    if road_condition == "Poor": acc_score += 25
    elif road_condition == "Potholed": acc_score += 40
    
    # Random noise (accidents happen unpredictably too)
    acc_score += random.randint(-10, 10)
    
    if acc_score > 75:
        accident_level = "High"
    elif acc_score > 40:
        accident_level = "Normal"
    else:
        accident_level = "Low"
        
    # Random edge cases to keep ML from overfitting a pure math rule
    if random.random() < 0.05:
        accident_level = random.choice(["Low", "Normal", "High"])

    data.append({
        "Zone": zone["name"],
        "Latitude": zone["lat"],
        "Longitude": zone["lon"],
        "Traffic": traffic,
        "Weather": weather,
        "Road_Condition": road_condition,
        "Pollution_AQI": aqi,
        "Water_Quality": water_quality,
        "Transport_Connection": transport_connection,
        "Accident_Prone_Level": accident_level
    })

df = pd.DataFrame(data)
df.to_csv("citysense_dataset.csv", index=False)
print(f"Dataset generated with {NUM_ROWS} rows => citysense_dataset.csv")
