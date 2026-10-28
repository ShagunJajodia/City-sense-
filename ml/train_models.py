import pandas as pd
import numpy as np
import time
import json
import joblib
import os
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("citysense_dataset.csv")

# Categorical mapping
categorical_cols = ["Zone", "Traffic", "Weather", "Road_Condition", "Water_Quality"]
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

target_le = LabelEncoder()
y = target_le.fit_transform(df["Accident_Prone_Level"].astype(str))
encoders["Accident_Prone_Level"] = target_le

X = df.drop(columns=["Accident_Prone_Level", "Latitude", "Longitude"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

metrics_data = {
    "models": [],
    "classes": target_le.classes_.tolist() # usually ['High', 'Low', 'Normal']
}

best_acc = 0
best_model = None

for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    # macro keeps it balanced multi-class
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, preds).tolist()
    
    train_time = round(end_time - start_time, 4)
    
    metrics_data["models"].append({
        "name": name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "training_time": train_time,
        "confusion_matrix": cm
    })
    
    if acc > best_acc:
        best_acc = acc
        best_model = model

os.makedirs("ml/models", exist_ok=True)
joblib.dump(best_model, "ml/models/best_model.pkl")
joblib.dump(encoders, "ml/models/encoders.pkl")

# SHAP XAI: Use Random Forest explainer regardless of "best" as it's guaranteed to be a Tree
explainer = shap.TreeExplainer(models["Random Forest"])
joblib.dump(explainer, "ml/models/explainer.pkl")

features_list = list(X.columns)
with open("ml/models/features.json", "w") as f:
    json.dump(features_list, f)

with open("ml/models/model_metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=4)

print(f"Models trained. Best was {type(best_model).__name__} with {best_acc:.4f} accuracy.")
print("Saved models, encoders, features, and model_metrics.json.")
