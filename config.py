"""
config.py – Central configuration for the Predictive Maintenance project.
All paths, model parameters, and feature definitions live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env file ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ── Dataset ─────────────────────────────────────────────────────────────────
DATASET_PATH = BASE_DIR / os.getenv("DATASET_PATH", "data/predictive_maintenance.csv")
MODEL_DIR = BASE_DIR / os.getenv("MODEL_DIR", "models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature definitions ─────────────────────────────────────────────────────
# Numerical sensor features
NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Categorical features (will be one-hot-encoded)
CATEGORICAL_FEATURES = ["Type"]

# All input features used by the models
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ── Target columns ──────────────────────────────────────────────────────────
# Actual Kaggle dataset uses "Target" (not "Machine failure")
TARGET_FAILURE = "Target"                   # Binary: 0 = Normal, 1 = Failure

# The dataset has a direct "Failure Type" string column (no separate binary cols)
# Individual binary indicator columns (present in some versions of the dataset)
FAILURE_TYPE_COLUMNS = [
    "TWF",   # Tool Wear Failure
    "HDF",   # Heat Dissipation Failure
    "PWF",   # Power Failure
    "OSF",   # Overstrain Failure
    "RNF",   # Random Failures
]

# The actual target column for failure type (direct string in this dataset)
TARGET_FAILURE_TYPE = "Failure Type"

# Failure type mapping – dataset uses full strings directly
FAILURE_TYPE_MAP = {
    "No Failure":               "No Failure",
    "Tool Wear Failure":        "Tool Wear Failure",
    "Heat Dissipation Failure": "Heat Dissipation Failure",
    "Power Failure":            "Power Failure",
    "Overstrain Failure":       "Overstrain Failure",
    "Random Failures":          "Random Failures",
    # Short codes (legacy / other dataset versions)
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failures",
    "Multiple": "Multiple Failures",
}

# ── Model files ─────────────────────────────────────────────────────────────
FAILURE_MODEL_PATH = MODEL_DIR / "failure_model.pkl"
FAILURE_TYPE_MODEL_PATH = MODEL_DIR / "failure_type_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"

# ── ML Parameters ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ── IBM Cloud ────────────────────────────────────────────────────────────────
IBM_API_KEY = os.getenv("IBM_API_KEY", "")
IBM_WML_URL = os.getenv("IBM_WML_URL", "https://us-south.ml.cloud.ibm.com")
IBM_WML_SPACE_ID = os.getenv("IBM_WML_SPACE_ID", "")
IBM_WML_DEPLOYMENT_ID = os.getenv("IBM_WML_DEPLOYMENT_ID", "")
IBM_WATSON_STUDIO_PROJECT_ID = os.getenv("IBM_WATSON_STUDIO_PROJECT_ID", "")
IBM_COS_API_KEY = os.getenv("IBM_COS_API_KEY", "")
IBM_COS_SERVICE_INSTANCE_ID = os.getenv("IBM_COS_SERVICE_INSTANCE_ID", "")
IBM_COS_ENDPOINT = os.getenv("IBM_COS_ENDPOINT", "https://s3.us-south.cloud-object-storage.appdomain.cloud")
IBM_COS_BUCKET = os.getenv("IBM_COS_BUCKET", "predictive-maintenance-models")
IBM_INTEGRATION_ENABLED = os.getenv("IBM_INTEGRATION_ENABLED", "false").lower() == "true"

# IBM Watson NLU
IBM_NLU_SERVICE_URL = os.getenv(
    "IBM_NLU_SERVICE_URL",
    "https://api.us-south.natural-language-understanding.watson.cloud.ibm.com"
)

# IBM watsonx.ai
IBM_WATSONX_URL = os.getenv("IBM_WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
IBM_WATSONX_MODEL_ID = os.getenv("IBM_WATSONX_MODEL_ID", "ibm/granite-13b-instruct-v2")

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Dashboard ─────────────────────────────────────────────────────────────────
APP_TITLE = "Predictive Maintenance of Industrial Machinery"
APP_ICON = "⚙️"
APP_VERSION = "1.0.0"

# Maintenance recommendation thresholds
TOOL_WEAR_WARNING_THRESHOLD = 180   # minutes
TOOL_WEAR_CRITICAL_THRESHOLD = 220  # minutes
TORQUE_WARNING_THRESHOLD = 60       # Nm
TEMP_DIFF_WARNING_THRESHOLD = 12    # K (process - air)
