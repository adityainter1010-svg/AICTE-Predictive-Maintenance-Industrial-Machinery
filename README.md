# ⚙️ Predictive Maintenance of Industrial Machinery
**AICTE Problem Statement #39 – Mechanical Engineering (Machine Learning Project)**

An AI/ML-based predictive maintenance system that anticipates industrial machine failures
before they occur using real-time sensor data, explainable AI (SHAP), and IBM Cloud Lite services.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Technology Stack](#technology-stack)
4. [Dataset Setup](#dataset-setup)
5. [Installation](#installation)
6. [IBM Cloud Lite Setup](#ibm-cloud-lite-setup)
7. [Training the Models](#training-the-models)
8. [Running the Dashboard](#running-the-dashboard)
9. [Running the Tests](#running-the-tests)
10. [Project Features](#project-features)
11. [Model Evaluation Results](#model-evaluation-results)
12. [IBM Cloud Integration Status](#ibm-cloud-integration-status)

---

## Project Overview

This system:
- **Predicts** whether an industrial machine will fail (binary classification)
- **Classifies** the failure type: Tool Wear (TWF), Heat Dissipation (HDF), Power Failure (PWF), Overstrain (OSF), Random (RNF)
- **Explains** predictions using SHAP (SHapley Additive exPlanations)
- **Recommends** targeted maintenance actions per failure category
- **Integrates** with IBM Watson Machine Learning (Lite) and IBM Cloud Object Storage (Lite)

---

## Folder Structure

```
predictive-maintenance/
│
├── app.py                          # Main Streamlit dashboard entry point
├── config.py                       # Central configuration (paths, features, IBM)
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (credentials – never commit)
├── README.md                       # This file
│
├── data/
│   └── predictive_maintenance.csv  # ← Place Kaggle dataset here
│
├── models/                         # Auto-created after training
│   ├── failure_model.pkl
│   ├── failure_type_model.pkl
│   ├── preprocessor.pkl
│   ├── label_encoder.pkl
│   ├── feature_names.pkl
│   └── training_summary.json
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Data loading, feature engineering, preprocessing
│   ├── train_model.py              # Model training (LR, DT, RF, XGBoost)
│   ├── prediction.py               # Real-time and batch prediction
│   ├── failure_classification.py   # Failure type helpers
│   ├── evaluation.py               # Metrics and plots
│   ├── explainability.py           # SHAP integration
│   ├── recommendations.py          # Maintenance recommendations
│   └── ibm_integration.py          # IBM WML + COS integration
│
├── pages/
│   ├── 1_Dataset_Analysis.py       # EDA page
│   ├── 2_Failure_Prediction.py     # Prediction input page
│   ├── 3_Model_Evaluation.py       # Model comparison page
│   └── 4_Explainable_AI.py         # SHAP page
│
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py
    └── test_prediction.py
```

---

## Technology Stack

| Category | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | Core development |
| ML | Scikit-learn, XGBoost | Classification models |
| Imbalance handling | imbalanced-learn (SMOTE) | Handle 3.4% failure rate |
| Explainability | SHAP | Feature importance & prediction explanation |
| Dashboard | Streamlit | Interactive web UI |
| Visualization | Plotly, Matplotlib, Seaborn | Charts and plots |
| Data | Pandas, NumPy | Data manipulation |
| IBM Cloud | Watson Machine Learning (Lite) | Cloud model deployment & scoring |
| IBM Cloud | Cloud Object Storage (Lite) | Remote model storage |
| Testing | pytest | Unit tests |
| Config | python-dotenv | Secure credential management |

---

## Dataset Setup

1. Go to [Kaggle Dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)
2. Download `predictive_maintenance.csv`
3. Place it in the `data/` folder:

```
predictive-maintenance/
└── data/
    └── predictive_maintenance.csv    ← here
```

The dataset contains:
- **10,000 records** of machine sensor readings
- **14 columns**: UDI, Product ID, Type, Air/Process Temperature, Rotational Speed, Torque, Tool Wear, Machine Failure flag, and 5 failure type indicators (TWF, HDF, PWF, OSF, RNF)

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone / navigate to project folder
cd predictive-maintenance

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## IBM Cloud Lite Setup

### IBM Services Used in This Project

| Service | Plan | Feature |
|---|---|---|
| **Watson Studio** | Lite (Free) | Project hosting, AutoAI, notebooks |
| **Watson Machine Learning** | Lite (Free) | REST model deployment + scoring |
| **Cloud Object Storage** | Lite (Free) | Remote model file storage |
| **Watson NLU** | Lite (Free) | Maintenance log text analysis |
| **watsonx.ai** | Lite (Free) | AI report generation (Granite model) |

**Active Credentials:**
- Watson Studio Project ID: `1b36f8d5-0bbb-4db1-a8e2-aecf9457aa5b`
- API Key: Configured in `.env`

### Step 1 – Create IBM Cloud Account
1. Visit https://cloud.ibm.com/registration
2. Sign up for a **free Lite account** (no credit card required)
3. Verify your email and log in

### Step 2 – Create Watson Studio Instance
1. Go to IBM Cloud Catalog → search for **Watson Studio**
2. Select **Lite plan** → Create
3. Click **Launch Watson Studio** → create a new **Project**
4. Note the **Project ID** from the project URL

### Step 3 – Create Watson Machine Learning Instance
1. IBM Cloud Catalog → **Watson Machine Learning** → Lite plan → Create
2. Go to **Service credentials** → **New credential** → Copy API Key
3. In Watson Studio project → **Settings** → associate the WML instance

### Step 4 – Create Watson NLU Instance (for log analysis)
1. IBM Cloud Catalog → **Natural Language Understanding** → Lite plan → Create
2. Go to **Service credentials** → **New credential** → note the service URL

### Step 5 – Deploy Model in Watson Studio
1. In your Watson Studio project → **Assets** → **Models**
2. Upload `failure_model.pkl` (or use AutoAI to train)
3. **Promote** the model → select a **Deployment Space**
4. Create an **Online Deployment** → copy the **Deployment ID** and **Space ID**

### Step 6 – Create Cloud Object Storage
1. IBM Cloud Catalog → **Object Storage** → Lite plan → Create
2. Create a bucket named `predictive-maintenance-models`
3. **Service credentials** → **New credential** → enable **HMAC** → copy keys

### Step 7 – Configure `.env`
Edit `.env` with your actual credentials:

```env
IBM_API_KEY=K_ZX1zGfBbKfBGzp4G5HqaTBUq1JJsXd_55CncC8YN8o
IBM_WML_URL=https://us-south.ml.cloud.ibm.com
IBM_WML_SPACE_ID=your_deployment_space_id
IBM_WML_DEPLOYMENT_ID=your_deployment_id
IBM_WATSON_STUDIO_PROJECT_ID=1b36f8d5-0bbb-4db1-a8e2-aecf9457aa5b
IBM_NLU_SERVICE_URL=https://api.us-south.natural-language-understanding.watson.cloud.ibm.com
IBM_COS_API_KEY=your_cos_api_key
IBM_COS_SERVICE_INSTANCE_ID=your_cos_service_instance_id
IBM_INTEGRATION_ENABLED=true
```

---

## Training the Models

After placing the dataset in `data/`, run:

```bash
cd predictive-maintenance
python -m src.train_model
```

This will:
1. Load and preprocess the dataset
2. Engineer features (Temp_Diff, Power_W, Wear_Category)
3. Apply SMOTE to handle class imbalance
4. Train 4 classifiers for both tasks (8 models total)
5. Select best by CV F1-macro
6. Save models to `models/`
7. Save training summary to `models/training_summary.json`

Expected training time: **1–3 minutes** on a typical laptop.

---

## Running the Dashboard

```bash
cd predictive-maintenance
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`

### Dashboard Pages

| Page | Description |
|---|---|
| **Home** | Project overview, KPIs, IBM status banner, log analyser |
| **1 – Dataset Analysis** | EDA, distributions, correlation, class balance |
| **2 – Failure Prediction** | Interactive sensor input → prediction + recommendations |
| **3 – Model Evaluation** | Metrics, confusion matrix, ROC curve, model comparison |
| **4 – Explainable AI** | SHAP summary, bar chart, waterfall plots |
| **5 – IBM Cloud** | IBM connection tests, WML deployment, COS upload, project info |
| **6 – Sensor Analytics** | Threshold monitoring, sensor trends, failure correlations |
| **7 – Maintenance Log Analyser** | IBM Watson NLP log analysis + watsonx.ai report generation |

---

## Running the Tests

```bash
cd predictive-maintenance
pytest tests/ -v
```

Expected output:
```
tests/test_preprocessing.py::TestInspectDataset::test_returns_dict         PASSED
tests/test_preprocessing.py::TestEngineerFeatures::test_temp_diff_created  PASSED
...
tests/test_prediction.py::TestGetRiskLevel::test_low_risk                   PASSED
...
PASSED 30+ tests
```

Tests cover:
- Dataset inspection
- Feature engineering (Temp_Diff, Power_W, Wear_Category)
- Failure type label derivation
- Preprocessing pipeline
- Risk level classification
- Input DataFrame construction
- Maintenance recommendations
- Failure classification helpers

---

## Project Features

### Machine Learning Pipeline
- **4 algorithms**: Logistic Regression, Decision Tree, Random Forest, XGBoost
- **2 tasks**: Binary failure detection + multi-class failure type
- **Imbalance handling**: SMOTE resampling + class weights
- **Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- **Cross-validation**: 5-fold stratified CV for fair comparison
- **No data leakage**: Preprocessor fitted only on training data

### Feature Engineering
| Feature | Formula | Purpose |
|---|---|---|
| `Temp_Diff` | Process Temp − Air Temp | Heat Dissipation Failure indicator |
| `Power_W` | Torque × rpm × 2π/60 | Power Failure detection |
| `Wear_Category` | Binned Tool Wear | Ordinal risk level |

### Explainable AI (SHAP)
- Global importance: beeswarm and bar charts
- Individual explanations: waterfall plots
- Top contributing features table

### Maintenance Recommendations
Each failure type triggers specific, actionable recommendations with urgency level and timeline.

---

## Model Evaluation Results

> **Note:** The actual results shown below are generated during training and may differ based on dataset split.
> Run `python -m src.train_model` and check `models/training_summary.json` for your actual results.
> The dashboard's Model Evaluation page shows live metrics from the held-out test set.

Typical performance on this dataset:

| Model | Task | CV F1-Macro |
|---|---|---|
| XGBoost | Machine Failure | ~0.92–0.96 |
| Random Forest | Machine Failure | ~0.90–0.94 |
| XGBoost | Failure Type | ~0.85–0.92 |
| Random Forest | Failure Type | ~0.82–0.90 |

Key metrics prioritised:
- **Recall** (failure detection sensitivity) – critical to minimize missed failures
- **F1-Macro** – balanced across imbalanced classes

---

## IBM Cloud Integration Status

| Component | Status | Notes |
|---|---|---|
| IBM Watson Machine Learning | ✅ Implemented | `src/ibm_integration.py` – WMLClient |
| IBM Cloud Object Storage | ✅ Implemented | `src/ibm_integration.py` – COSClient |
| IAM Authentication | ✅ Implemented | Token exchange via IAM endpoint |
| Model Deployment Scoring | ✅ Implemented | REST API v4 scoring endpoint |
| COS Model Upload/Download | ✅ Implemented | PUT/GET object storage |
| Dashboard IBM Status | ✅ Shown in sidebar | Connection test button |
| Credentials Security | ✅ `.env` / env vars | Never hardcoded |
| **Active IBM Connection** | ⚙️ Requires setup | Configure credentials in `.env` |

To activate IBM integration:
1. Complete the [IBM Cloud Lite Setup](#ibm-cloud-lite-setup) steps
2. Set `IBM_INTEGRATION_ENABLED=true` in `.env`
3. Fill in all `IBM_*` variables with real credentials

---

## Contact / Attribution

- **Project**: AICTE Internship Problem Statement #39
- **Domain**: Mechanical Engineering – Machine Learning
- **Dataset**: [Kaggle – Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)
- **IBM Cloud**: [https://cloud.ibm.com](https://cloud.ibm.com)
