# Import all required libraries
import pandas as pd
import numpy as np
import json
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# ML & Explainability
from sklearn.preprocessing import LabelEncoder
import shap

# API Integration
import urllib.request

# Set styles
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("✅ All dependencies loaded successfully!")

# Define paths
base_dir = os.path.abspath('.')
model_dir = os.path.join(base_dir, 'ml', 'models')

# Load models
model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
encoders = joblib.load(os.path.join(model_dir, 'encoders.pkl'))
explainer = joblib.load(os.path.join(model_dir, 'explainer.pkl'))

# Load features list
with open(os.path.join(model_dir, 'features.json'), 'r') as f:
    features_list = json.load(f)

# Load model metrics
with open(os.path.join(model_dir, 'model_metrics.json'), 'r') as f:
    model_metrics = json.load(f)

print(f"✅ Best Model loaded: {type(model).__name__}")
print(f"✅ Encoders loaded for: {list(encoders.keys())}")
print(f"✅ Features ({len(features_list)}): {features_list}")

# Bengaluru zones with their baseline characteristics
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
    {"name": "Jayanagar", "lat": 12.9299, "lon": 77.5826, "base_aqi": 60, "base_water": "Good", "base_traffic": "Normal", "transport": 8},
    {"name": "HSR Layout", "lat": 12.9121, "lon": 77.6446, "base_aqi": 85, "base_water": "Good", "base_traffic": "High", "transport": 8},
]

zones_df = pd.DataFrame(BENGALURU_ZONES)
print(f"✅ Loaded {len(zones_df)} Bengaluru zones")
zones_df.head()

def make_prediction(zone_data):
    """
    Make a prediction for given input parameters.
    Returns prediction with SHAP explanation and livability index.
    """
    # Create DataFrame from input
    df = pd.DataFrame([zone_data])
    
    # Encode categorical variables
    categorical_cols = ["Zone", "Traffic", "Weather", "Road_Condition", "Water_Quality"]
    for col in categorical_cols:
        if zone_data[col] in encoders[col].classes_:
            df[col] = encoders[col].transform([zone_data[col]])
        else:
            df[col] = encoders[col].transform([encoders[col].classes_[0]])
    
    # Select features in correct order
    df = df[features_list]
    
    # Make prediction
    pred_idx = int(model.predict(df)[0])
    pred_label = encoders["Accident_Prone_Level"].inverse_transform([pred_idx])[0]
    pred_proba = model.predict_proba(df)[0]
    
    # SHAP Explainable AI
    xai_reason = "Unable to compute feature importance."
    try:
        shap_vals = explainer.shap_values(df)
        if isinstance(shap_vals, list): 
            class_shap = shap_vals[pred_idx][0]
        else:
            if len(shap_vals.shape) == 3:
                class_shap = shap_vals[0, :, pred_idx]
            else:
                class_shap = shap_vals[0]
        
        max_idx = np.argmax(np.abs(class_shap))
        top_feature = features_list[max_idx]
        top_impact = class_shap[max_idx]
        impact_dir = "increased" if top_impact > 0 else "decreased"
        xai_reason = f"Top influencer: {top_feature.replace('_', ' ')} ({abs(top_impact)*100:.1f}% impact)"
    except Exception as e:
        pass
    
    # Calculate Livability Index
    livability = 100
    pollution_aqi = zone_data['Pollution_AQI']
    if pollution_aqi > 50: 
        livability -= min(30, (pollution_aqi-50)/5)
    if zone_data['Traffic'] == "Severe": 
        livability -= 20
    elif zone_data['Traffic'] == "High": 
        livability -= 10
    if zone_data['Water_Quality'] == "Poor": 
        livability -= 15
    elif zone_data['Water_Quality'] == "Fair": 
        livability -= 5
    livability += (zone_data['Transport_Connection'] - 5) * 2
    livability = max(0, min(100, livability))
    
    return {
        'accident_level': pred_label,
        'accident_proba': dict(zip(encoders["Accident_Prone_Level"].classes_, pred_proba)),
        'livability_index': round(livability, 1),
        'xai_reason': xai_reason
    }

print("✅ Prediction function defined!")

# Make predictions for different zones and conditions
test_scenarios = [
    # Scenario 1: High-risk zone during rush hour
    {
        'Zone': 'Whitefield',
        'Traffic': 'Severe',
        'Weather': 'Rain',
        'Road_Condition': 'Potholed',
        'Pollution_AQI': 250,
        'Water_Quality': 'Fair',
        'Transport_Connection': 8
    },
    # Scenario 2: Safe zone, normal conditions
    {
        'Zone': 'Malleswaram',
        'Traffic': 'Low',
        'Weather': 'Clear',
        'Road_Condition': 'Excellent',
        'Pollution_AQI': 60,
        'Water_Quality': 'Good',
        'Transport_Connection': 8
    },
    # Scenario 3: Medium-risk zone
    {
        'Zone': 'Koramangala',
        'Traffic': 'High',
        'Weather': 'Heavy Rain',
        'Road_Condition': 'Good',
        'Pollution_AQI': 130,
        'Water_Quality': 'Good',
        'Transport_Connection': 9
    },
]

predictions = []
for i, scenario in enumerate(test_scenarios, 1):
    pred = make_prediction(scenario)
    pred['scenario'] = f"Scenario {i}"
    pred['zone'] = scenario['Zone']
    predictions.append(pred)
    print(f"\n{pred['scenario']} - {pred['zone']}")
    print(f"  Accident Level: {pred['accident_level']}")
    print(f"  Livability Index: {pred['livability_index']}/100")
    print(f"  🧠 XAI: {pred['xai_reason']}")

# Create comparison visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accident Risk Distribution
scenarios = [p['scenario'] for p in predictions]
accident_levels = [p['accident_level'] for p in predictions]
colors = ['#ef4444' if level == 'High' else '#f59e0b' if level == 'Normal' else '#10b981' 
          for level in accident_levels]

axes[0].bar(scenarios, accident_levels, color=colors, alpha=0.7, edgecolor='black')
axes[0].set_ylabel('Accident Risk Level')
axes[0].set_title('Accident Risk by Scenario')
axes[0].set_ylim(0, max([len(encoders["Accident_Prone_Level"].classes_), 3]))

# Livability Index
livability = [p['livability_index'] for p in predictions]
colors_liv = ['#10b981' if liv > 70 else '#f59e0b' if liv > 50 else '#ef4444' 
              for liv in livability]

axes[1].bar(scenarios, livability, color=colors_liv, alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Livability Index')
axes[1].set_title('Livability Index by Scenario')
axes[1].set_ylim(0, 100)
axes[1].axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Good')
axes[1].axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Acceptable')
axes[1].legend()

plt.tight_layout()
pass

print("✅ Prediction visualizations complete!")

# Display model comparison metrics
metrics_df = pd.DataFrame(model_metrics['models'])
print("\n📊 Model Performance Comparison:")
print(metrics_df[['name', 'accuracy', 'precision', 'recall', 'f1', 'training_time']].to_string(index=False))

print(f"\n🏆 Best Model: {model_metrics['models'][0]['name']} (Accuracy: {model_metrics['models'][0]['accuracy']:.2%})")

# Visualize model comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Accuracy
axes[0, 0].bar(metrics_df['name'], metrics_df['accuracy'], color='skyblue', edgecolor='black')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Model Accuracy')
axes[0, 0].set_ylim([0, 1])

# Precision, Recall, F1
x_pos = np.arange(len(metrics_df))
width = 0.25
axes[0, 1].bar(x_pos - width, metrics_df['precision'], width, label='Precision', color='lightgreen', edgecolor='black')
axes[0, 1].bar(x_pos, metrics_df['recall'], width, label='Recall', color='lightyellow', edgecolor='black')
axes[0, 1].bar(x_pos + width, metrics_df['f1'], width, label='F1', color='lightcoral', edgecolor='black')
axes[0, 1].set_ylabel('Score')
axes[0, 1].set_title('Precision, Recall, F1')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(metrics_df['name'], rotation=15, ha='right')
axes[0, 1].legend()
axes[0, 1].set_ylim([0, 1])

# Training Time
axes[1, 0].bar(metrics_df['name'], metrics_df['training_time'], color='lightsteelblue', edgecolor='black')
axes[1, 0].set_ylabel('Time (seconds)')
axes[1, 0].set_title('Training Time')
axes[1, 0].tick_params(axis='x', rotation=15)

# Confusion Matrix of Best Model
cm = np.array(metrics_df.loc[0, 'confusion_matrix'])
im = axes[1, 1].imshow(cm, cmap='Blues', aspect='auto')
axes[1, 1].set_title('Confusion Matrix (Best Model)')
axes[1, 1].set_ylabel('True Label')
axes[1, 1].set_xlabel('Predicted Label')
for i in range(len(cm)):
    for j in range(len(cm)):
        axes[1, 1].text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
pass

def get_traffic_for_time(base_traffic, hour):
    """
    Simulate traffic conditions based on time of day.
    """
    hour = int(hour)
    # Rush hours: 8-10 AM, 5-8 PM (17-20)
    is_rush = (hour >= 8 and hour <= 10) or (hour >= 17 and hour <= 20)
    is_night = (hour <= 5 or hour >= 23)
    
    if is_rush:
        if base_traffic == "Low": return "Normal"
        if base_traffic == "Normal": return "High"
        return "Severe"
    if is_night:
        if base_traffic == "Severe" or base_traffic == "High": return "Normal"
        return "Low"
    return base_traffic

# Simulate predictions throughout the day for a single zone
test_zone = zones_df.iloc[0].to_dict()  # Whitefield
hours = list(range(0, 24, 2))
risk_levels = []
livability_scores = []

for hour in hours:
    simulated_traffic = get_traffic_for_time(test_zone['base_traffic'], hour)
    
    scenario = {
        'Zone': test_zone['name'],
        'Traffic': simulated_traffic,
        'Weather': 'Clear',
        'Road_Condition': 'Good',
        'Pollution_AQI': test_zone['base_aqi'],
        'Water_Quality': test_zone['base_water'],
        'Transport_Connection': test_zone['transport']
    }
    
    pred = make_prediction(scenario)
    risk_level_num = {'High': 3, 'Normal': 2, 'Low': 1}.get(pred['accident_level'], 1)
    risk_levels.append(risk_level_num)
    livability_scores.append(pred['livability_index'])

# Visualize time-based changes
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Risk Level over time
axes[0].plot(hours, risk_levels, marker='o', linewidth=2, markersize=8, color='#ef4444')
axes[0].fill_between(hours, risk_levels, alpha=0.3, color='#ef4444')
axes[0].set_xlabel('Hour of Day')
axes[0].set_ylabel('Accident Risk Level')
axes[0].set_title(f'Accident Risk Throughout Day - {test_zone["name"]}')
axes[0].set_xticks(hours)
axes[0].set_xticklabels([f'{h:02d}:00' for h in hours], rotation=45)
axes[0].grid(True, alpha=0.3)

# Livability Score over time
axes[1].plot(hours, livability_scores, marker='s', linewidth=2, markersize=8, color='#10b981')
axes[1].fill_between(hours, livability_scores, alpha=0.3, color='#10b981')
axes[1].set_xlabel('Hour of Day')
axes[1].set_ylabel('Livability Index')
axes[1].set_title(f'Livability Throughout Day - {test_zone["name"]}')
axes[1].set_xticks(hours)
axes[1].set_xticklabels([f'{h:02d}:00' for h in hours], rotation=45)
axes[1].set_ylim([0, 100])
axes[1].axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Good')
axes[1].axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Acceptable')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
pass

# Predict for all zones at noon time
all_zones_predictions = []

for _, zone in zones_df.iterrows():
    scenario = {
        'Zone': zone['name'],
        'Traffic': zone['base_traffic'],
        'Weather': 'Clear',
        'Road_Condition': 'Good',
        'Pollution_AQI': zone['base_aqi'],
        'Water_Quality': zone['base_water'],
        'Transport_Connection': zone['transport']
    }
    
    pred = make_prediction(scenario)
    all_zones_predictions.append({
        'zone': zone['name'],
        'lat': zone['lat'],
        'lon': zone['lon'],
        'base_aqi': zone['base_aqi'],
        'accident_level': pred['accident_level'],
        'livability_index': pred['livability_index']
    })

zones_pred_df = pd.DataFrame(all_zones_predictions)
print("\n📍 All Zones Predictions (Noon Time):")
print(zones_pred_df[['zone', 'accident_level', 'livability_index', 'base_aqi']].to_string(index=False))

# Create comprehensive zone visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Risk Distribution by Zone
risk_counts = zones_pred_df['accident_level'].value_counts()
colors_pie = {'High': '#ef4444', 'Normal': '#f59e0b', 'Low': '#10b981'}
colors = [colors_pie.get(level, '#gray') for level in risk_counts.index]

axes[0, 0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
               colors=colors, startangle=90)
axes[0, 0].set_title('Risk Distribution Across Zones')

# Livability Index by Zone (sorted)
zones_sorted = zones_pred_df.sort_values('livability_index', ascending=True)
bars = axes[0, 1].barh(zones_sorted['zone'], zones_sorted['livability_index'], 
                         color=['#10b981' if x > 70 else '#f59e0b' if x > 50 else '#ef4444' 
                                for x in zones_sorted['livability_index']], 
                         edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Livability Index')
axes[0, 1].set_title('Livability Index by Zone')
axes[0, 1].set_xlim([0, 100])

# AQI vs Livability
colors_scatter = [colors_pie.get(level, '#gray') for level in zones_pred_df['accident_level']]
axes[1, 0].scatter(zones_pred_df['base_aqi'], zones_pred_df['livability_index'], 
                   s=200, c=colors_scatter, alpha=0.6, edgecolors='black')
axes[1, 0].set_xlabel('Pollution AQI')
axes[1, 0].set_ylabel('Livability Index')
axes[1, 0].set_title('Air Quality vs Livability Index')
axes[1, 0].grid(True, alpha=0.3)

# Accident Level Distribution
zone_names_short = [z[:12] for z in zones_pred_df['zone'].values]
risk_numeric = {'High': 3, 'Normal': 2, 'Low': 1}.values()
risk_numeric = [{'High': 3, 'Normal': 2, 'Low': 1}[level] for level in zones_pred_df['accident_level']]
bars = axes[1, 1].bar(range(len(zones_pred_df)), risk_numeric, 
                       color=[colors_pie.get(level, '#gray') for level in zones_pred_df['accident_level']], 
                       edgecolor='black', alpha=0.7)
axes[1, 1].set_ylabel('Risk Level (1=Low, 2=Normal, 3=High)')
axes[1, 1].set_title('Accident Risk by Zone')
axes[1, 1].set_xticks(range(len(zones_pred_df)))
axes[1, 1].set_xticklabels(zone_names_short, rotation=45, ha='right')
axes[1, 1].set_ylim([0, 3.5])

plt.tight_layout()
pass

print("✅ Zone analysis visualizations complete!")

import json

def call_api_endpoint(endpoint, method='GET', data=None):
    """
    Helper function to call the CitySense API endpoints.
    """
    url = f'http://127.0.0.1:8000{endpoint}'
    try:
        if method == 'GET':
            with urllib.request.urlopen(url, timeout=5) as response:
                return json.loads(response.read().decode())
        elif method == 'POST':
            req = urllib.request.Request(url, data=json.dumps(data).encode(), 
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=5) as response:
                return json.loads(response.read().decode())
    except Exception as e:
        return {'error': str(e)}

# Test API endpoints
print("🔌 Testing CitySense API Endpoints\n")

# Test /zones endpoint
try:
    zones_api = call_api_endpoint('/zones')
    print(f"✅ /zones: Retrieved {len(zones_api.get('zones', []))} zones")
except:
    print("❌ /zones: Failed to connect")

# Test /metrics endpoint
try:
    metrics_api = call_api_endpoint('/metrics')
    if 'models' in metrics_api:
        print(f"✅ /metrics: Retrieved metrics for {len(metrics_api['models'])} models")
        for model_info in metrics_api['models']:
            print(f"   - {model_info['name']}: Accuracy={model_info['accuracy']:.2%}")
except:
    print("❌ /metrics: Failed to connect")

# Test /predict endpoint
try:
    test_input = {
        'Zone': 'Electronic City',
        'Traffic': 'High',
        'Weather': 'Rain',
        'Road_Condition': 'Average',
        'Pollution_AQI': 170,
        'Water_Quality': 'Fair',
        'Transport_Connection': 7
    }
    api_prediction = call_api_endpoint('/predict', method='POST', data=test_input)
    print(f"\n✅ /predict: Prediction successful")
    print(f"   Accident Level: {api_prediction.get('accident_level')}")
    print(f"   Livability Index: {api_prediction.get('livability_index')}")
    print(f"   XAI Reason: {api_prediction.get('xai_reason')}")
except:
    print("❌ /predict: Failed to connect")

print("\n" + "="*60)
print("🎯 CitySense Project Summary")
print("="*60)

print("\n📊 PROJECT COMPONENTS:")
print("   1. Backend API (FastAPI) - REST & WebSocket endpoints")
print("   2. ML Models - Predicting accident risk levels")
print("   3. Explainability - SHAP-based feature importance")
print("   4. Urban Analytics - Livability index calculations")
print("   5. Frontend Dashboard - Interactive Leaflet map")
print("   6. Time Simulation - Traffic pattern modeling")

print("\n🤖 MODEL CAPABILITIES:")
best_model_info = model_metrics['models'][0]
print(f"   Model: {best_model_info['name']}")
print(f"   Accuracy: {best_model_info['accuracy']:.2%}")
print(f"   Precision: {best_model_info['precision']:.2%}")
print(f"   Recall: {best_model_info['recall']:.2%}")
print(f"   F1 Score: {best_model_info['f1']:.2%}")

print("\n🌍 COVERAGE:")
print(f"   Total Zones: {len(zones_df)}")
print(f"   High Risk Zones: {(zones_pred_df['accident_level']=='High').sum()}")
print(f"   Normal Risk Zones: {(zones_pred_df['accident_level']=='Normal').sum()}")
print(f"   Low Risk Zones: {(zones_pred_df['accident_level']=='Low').sum()}")

print(f"\n📈 METRICS:")
print(f"   Avg Livability Index: {zones_pred_df['livability_index'].mean():.1f}/100")
print(f"   Avg Pollution AQI: {zones_pred_df['base_aqi'].mean():.0f}")
print(f"   Highest Risk Zone: {zones_pred_df.loc[zones_pred_df['livability_index'].idxmin(), 'zone']}")
print(f"   Safest Zone: {zones_pred_df.loc[zones_pred_df['livability_index'].idxmax(), 'zone']}")

print("\n🔌 API ENDPOINTS:")
print("   GET /zones - List all monitored zones")
print("   GET /metrics - Model performance metrics")
print("   POST /predict - Make risk predictions")
print("   WS /ws/live_simulate - Live data streaming")

print("\n" + "="*60)
print("✨ CitySense: Urban Intelligence at Scale")
print("="*60)
