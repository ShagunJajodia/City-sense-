# CitySense: Urban Risk and Livability Classification in Bengaluru

CitySense is an IoT-driven and Explainable AI (XAI) framework designed to assess and classify urban risk and livability states across Bengaluru zones. By fusing environmental and infrastructural telemetry (traffic density, air quality, weather conditions, road integrity, and water quality), the system classifies zones into **High Risk**, **Normal Risk**, and **Low Risk**, backed by SHAP (SHapley Additive exPlanations) for municipal decision transparency.

---

## 🌟 Features

- **Multi-Modal Urban Telemetry:** Fuses air quality, traffic metrics, weather, and infrastructural data across 35 Bengaluru zones.
- **Machine Learning Classifiers:** Evaluates Random Forest, Decision Tree, and Logistic Regression models.
- **Explainable AI (SHAP):** Provides local feature attributions for every prediction to ensure transparent, accountable risk scoring.
- **FastAPI REST Backend:** High-performance API serving real-time predictions and SHAP explanations.
- **Interactive Web Dashboard:** Modern frontend interface with zone analytics and interactive maps.
- **Research Documentation:** Includes IEEE research paper (`Shagun_IEEE_Final_Version.pdf`) and LaTeX report source (`citysense_report.tex`).

---

## 📁 Repository Structure

```text
├── backend/
│   └── app.py                      # FastAPI REST API application
├── frontend/
│   ├── index.html                  # Web application UI
│   ├── script.js                   # Client logic and visualization
│   ├── style.css                   # Stylesheet
│   └── assets/                     # Frontend media and icons
├── ml/
│   ├── generate_data.py            # Synthetic IoT telemetry generator
│   ├── train_models.py             # Model training and SHAP pipeline
│   └── models/
│       ├── best_model.pkl          # Trained classifier model
│       ├── encoders.pkl            # Preprocessing encoders
│       ├── explainer.pkl           # SHAP explainer
│       ├── features.json           # Feature schema definition
│       └── model_metrics.json      # Model performance benchmarks
├── bengaluru_risk_data.csv         # Zone-specific baseline risk data
├── citysense_dataset.csv           # Synthesized multi-modal dataset
├── CitySense_Complete_Guide.ipynb  # Interactive walkthrough notebook
├── CitySense_Executed.ipynb        # Pre-executed experimental notebook
├── CitySense_Complete_Guide.py     # Python script export of notebook
├── Shagun_IEEE_Final_Version.pdf   # IEEE Conference Paper
└── citysense_report.tex            # LaTeX source for the research paper
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Modern Web Browser

### 2. Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install fastapi uvicorn scikit-learn pandas numpy shap joblib
   ```
3. Run the FastAPI server:
   ```bash
   uvicorn app:app --reload --port 8000
   ```

### 3. Frontend Setup
Simply open `frontend/index.html` in your web browser, or serve it using any static server (e.g., `python -m http.server 3000` inside `frontend/`).

---

## 👥 Authors
- **Shagun Jajodia**
- **Vivek Kumar Dubey**
- *School of Computer Engineering, Manipal Institute of Technology, Manipal, India*

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
