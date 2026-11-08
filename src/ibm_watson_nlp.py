"""
src/ibm_watson_nlp.py
IBM Watson NLP integration – analyses free-text maintenance log entries
to extract failure keywords, severity, and recommended actions.

Uses IBM Watson Natural Language Understanding (NLU) or
IBM watsonx.ai text generation via the watsonx REST API.
Both are available on IBM Cloud Lite / free tiers.

Authentication: IBM IAM bearer token (derived from IBM_API_KEY).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import requests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.ibm_integration import get_iam_token

logger = logging.getLogger(__name__)

# ── IBM NLU endpoint ─────────────────────────────────────────────────────────
# Watson NLU Lite plan – 30,000 NLU items/month free
NLU_BASE_URL = "https://api.us-south.natural-language-understanding.watson.cloud.ibm.com"
NLU_VERSION  = "2022-04-07"

# ── IBM watsonx.ai (Text Generation) endpoint ────────────────────────────────
WATSONX_BASE_URL  = "https://us-south.ml.cloud.ibm.com"
WATSONX_MODEL_ID  = "ibm/granite-13b-instruct-v2"   # IBM Granite – free tier

# ── Failure keyword lexicon ───────────────────────────────────────────────────
FAILURE_KEYWORDS = {
    "TWF": ["tool wear", "worn tool", "tool replacement", "cutting tool", "tool life",
            "tool broken", "worn insert", "worn cutter"],
    "HDF": ["overheating", "heat dissipation", "coolant", "temperature high",
            "thermal", "cooling failure", "hot", "heat buildup"],
    "PWF": ["power failure", "electrical fault", "voltage drop", "power surge",
            "motor current", "amperage", "power anomaly", "electrical"],
    "OSF": ["overstrain", "overload", "excessive torque", "mechanical overload",
            "strain", "excessive force", "overloaded"],
    "RNF": ["random failure", "unexpected shutdown", "unplanned stoppage",
            "intermittent fault", "random", "unclassified"],
}

SEVERITY_KEYWORDS = {
    "critical": ["immediate", "stop", "shutdown", "emergency", "critical", "urgent", "broken"],
    "warning":  ["check", "inspect", "monitor", "warning", "high", "elevated", "abnormal"],
    "info":     ["note", "routine", "scheduled", "normal", "regular", "standard"],
}


# ── Local Keyword-Based Analysis (no API required) ───────────────────────────

def analyse_log_local(log_text: str) -> dict:
    """
    Analyse a maintenance log entry using local keyword matching.
    Returns failure type candidates, severity, and recommendations.
    No IBM API call required.
    """
    text_lower = log_text.lower()
    detected_types = []
    type_scores = {}

    for ftype, keywords in FAILURE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            detected_types.append(ftype)
            type_scores[ftype] = score

    # Severity detection
    severity = "info"
    for level in ("critical", "warning", "info"):
        if any(kw in text_lower for kw in SEVERITY_KEYWORDS[level]):
            severity = level
            break

    # Top predicted failure type
    predicted_type = max(type_scores, key=type_scores.get) if type_scores else "No Failure"
    predicted_label = config.FAILURE_TYPE_MAP.get(predicted_type, predicted_type)

    # Generate recommendations
    from src.recommendations import get_recommendations
    rec = get_recommendations(predicted_type)

    return {
        "log_text": log_text,
        "detected_failure_types": [config.FAILURE_TYPE_MAP.get(t, t) for t in detected_types],
        "predicted_type_code": predicted_type,
        "predicted_type_label": predicted_label,
        "type_scores": {config.FAILURE_TYPE_MAP.get(k, k): v for k, v in type_scores.items()},
        "severity": severity,
        "urgency": rec.get("urgency", "N/A"),
        "recommendations": rec.get("actions", []),
        "analysis_method": "local_keyword",
    }


# ── IBM Watson NLU Integration ────────────────────────────────────────────────

class WatsonNLUClient:
    """
    IBM Watson Natural Language Understanding client.
    Lite plan: 30,000 NLU items/month free.
    Extracts entities, keywords, sentiment, and categories from maintenance logs.
    """

    def __init__(self, service_url: str = None, api_key: str = None):
        self.service_url = (service_url or NLU_BASE_URL).rstrip("/")
        self.api_key     = api_key or config.IBM_API_KEY
        self._token: str | None = None

    def _auth(self) -> dict:
        if not self._token:
            self._token = get_iam_token(self.api_key)
        return {"Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json"}

    def analyse(self, text: str, features: dict = None) -> dict:
        """
        Call Watson NLU /analyze endpoint.

        Default features: entities, keywords, sentiment, categories.
        Returns the raw NLU response dict.
        """
        if features is None:
            features = {
                "entities":   {"sentiment": True, "emotion": True, "limit": 10},
                "keywords":   {"sentiment": True, "emotion": True, "limit": 15},
                "sentiment":  {"document": True},
                "categories": {"limit": 3},
            }
        payload = {
            "text": text,
            "features": features,
            "language": "en",
        }
        url = f"{self.service_url}/v1/analyze?version={NLU_VERSION}"
        resp = requests.post(url, json=payload, headers=self._auth(), timeout=30)
        resp.raise_for_status()
        return resp.json()

    def extract_maintenance_insights(self, text: str) -> dict:
        """
        High-level wrapper: calls NLU and reformats the response for the dashboard.
        Returns a structured insights dict.
        """
        raw = self.analyse(text)

        # Extract top keywords
        keywords = [kw["text"] for kw in raw.get("keywords", [])[:8]]

        # Extract entities (equipment names, locations, etc.)
        entities = [
            {"text": e["text"], "type": e["type"], "relevance": round(e.get("relevance", 0), 3)}
            for e in raw.get("entities", [])
        ]

        # Document sentiment
        sentiment = raw.get("sentiment", {}).get("document", {})

        # Categories
        categories = [
            {"label": c["label"].strip("/"), "score": round(c["score"], 3)}
            for c in raw.get("categories", [])
        ]

        # Local failure type matching on top of NLU keywords
        enriched_text = text + " " + " ".join(keywords)
        local = analyse_log_local(enriched_text)

        return {
            "nlu_keywords": keywords,
            "nlu_entities": entities,
            "nlu_sentiment": sentiment,
            "nlu_categories": categories,
            "detected_failure_types": local["detected_failure_types"],
            "predicted_type_code":   local["predicted_type_code"],
            "predicted_type_label":  local["predicted_type_label"],
            "severity": local["severity"],
            "urgency":  local["urgency"],
            "recommendations": local["recommendations"],
            "analysis_method": "ibm_watson_nlu",
            "raw_nlu": raw,
        }


# ── IBM watsonx.ai Text Generation ───────────────────────────────────────────

class WatsonxAIClient:
    """
    IBM watsonx.ai (Granite model) – generate maintenance summaries and action plans.
    Uses IBM Granite-13B-Instruct (available on watsonx.ai Lite).
    """

    def __init__(self):
        self._token: str | None = None

    def _auth(self) -> dict:
        if not self._token:
            self._token = get_iam_token()
        return {"Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json"}

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        """
        Call watsonx.ai /text/generation endpoint.
        Returns the generated text string.
        """
        project_id = config.IBM_WATSON_STUDIO_PROJECT_ID
        if not project_id:
            raise ValueError("IBM_WATSON_STUDIO_PROJECT_ID not configured in .env")

        url = f"{WATSONX_BASE_URL}/ml/v1/text/generation?version=2023-05-29"
        payload = {
            "model_id": WATSONX_MODEL_ID,
            "project_id": project_id,
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": max_tokens,
                "stop_sequences": ["\n\n"],
                "temperature": 0.3,
            },
        }
        resp = requests.post(url, json=payload, headers=self._auth(), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [{}])[0].get("generated_text", "").strip()

    def summarise_maintenance_log(self, log_text: str) -> str:
        """Generate a concise maintenance action summary from a log entry."""
        prompt = (
            "You are an industrial maintenance expert AI.\n"
            f"Maintenance log:\n{log_text}\n\n"
            "Provide a brief summary of:\n"
            "1. Likely failure cause\n"
            "2. Immediate action required\n"
            "3. Preventive measure\n"
            "Summary:"
        )
        return self.generate(prompt, max_tokens=200)

    def generate_maintenance_report(self, failure_type: str, sensor_values: dict) -> str:
        """Generate a maintenance report for a given failure type and sensor readings."""
        sensor_text = "\n".join([f"  - {k}: {v}" for k, v in sensor_values.items()])
        prompt = (
            f"Industrial machine sensor readings:\n{sensor_text}\n"
            f"Predicted failure type: {failure_type}\n\n"
            "Write a professional maintenance report with:\n"
            "1. Root cause analysis\n"
            "2. Risk assessment\n"
            "3. Recommended corrective actions\n"
            "Report:"
        )
        return self.generate(prompt, max_tokens=350)


# ── High-level analysis function ─────────────────────────────────────────────

def analyse_maintenance_log(log_text: str, use_ibm_nlu: bool = False) -> dict:
    """
    Analyse a maintenance log entry.
    If IBM integration is enabled and use_ibm_nlu=True, calls Watson NLU.
    Otherwise falls back to local keyword analysis.
    """
    if config.IBM_INTEGRATION_ENABLED and use_ibm_nlu:
        try:
            client = WatsonNLUClient()
            return client.extract_maintenance_insights(log_text)
        except Exception as exc:
            logger.warning("Watson NLU call failed (%s). Falling back to local analysis.", exc)

    return analyse_log_local(log_text)


def generate_report_with_watsonx(failure_type: str, sensor_values: dict) -> str:
    """Generate a maintenance report using IBM watsonx.ai Granite model."""
    if not config.IBM_INTEGRATION_ENABLED:
        return "IBM watsonx.ai integration is disabled. Set IBM_INTEGRATION_ENABLED=true."
    if not config.IBM_WATSON_STUDIO_PROJECT_ID:
        return "IBM Watson Studio Project ID not configured."
    try:
        client = WatsonxAIClient()
        return client.generate_maintenance_report(failure_type, sensor_values)
    except Exception as exc:
        logger.error("watsonx.ai generation failed: %s", exc)
        return f"watsonx.ai generation failed: {exc}"


# ── Sample maintenance logs for demonstration ─────────────────────────────────
SAMPLE_LOGS = [
    "Machine #A12 experienced overheating during afternoon shift. Coolant flow rate dropped significantly. Thermal sensors showed abnormal readings above 320K. Immediate inspection of cooling system required.",
    "Tool holder on CNC unit 3 showing excessive vibration. Cutting tool appears worn beyond acceptable limits. Surface finish quality has deteriorated on last 50 parts. Tool replacement scheduled.",
    "Unexpected electrical fault on production line 2. Motor current spiked to 18A. Circuit breaker tripped. Power supply unit inspection needed before restart.",
    "Spindle torque readings exceeded 72 Nm during heavy cutting pass. Overstrain alert triggered. Reduced feed rate recommended to prevent mechanical failure.",
    "Routine inspection completed on machine B07. All parameters within normal range. Lubrication levels adequate. No immediate action required.",
]
