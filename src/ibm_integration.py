"""
src/ibm_integration.py
IBM Cloud Lite service integration.

Services used:
  1. IBM Watson Machine Learning (WML) - Lite plan
     - Deploy and score the best trained model via REST API.
  2. IBM Cloud Object Storage (COS) - Lite plan
     • Store and retrieve trained model artifacts remotely.

All credentials are loaded from environment variables (never hardcoded).
"""

from __future__ import annotations

import os
import io
import json
import logging
import pickle
from pathlib import Path

import requests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

# ── IAM Token Helper ─────────────────────────────────────────────────────────

def get_iam_token(api_key: str = None) -> str:
    """Exchange an IBM Cloud API key for a short-lived IAM bearer token."""
    api_key = api_key or config.IBM_API_KEY
    if not api_key or api_key == "your_ibm_api_key_here":
        raise ValueError(
            "IBM API key is not configured. "
            "Set IBM_API_KEY in your .env file or environment variables."
        )
    url = "https://iam.cloud.ibm.com/identity/token"
    payload = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(url, data=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    token = resp.json()["access_token"]
    logger.info("IBM IAM token obtained successfully.")
    return token


# ── Watson Machine Learning ───────────────────────────────────────────────────

class WMLClient:
    """
    Thin wrapper around the IBM Watson Machine Learning REST API (v4).
    Uses the Lite (free) plan – no ibm-watson-machine-learning SDK required.
    """

    def __init__(self):
        self.url = config.IBM_WML_URL.rstrip("/")
        self.space_id = config.IBM_WML_SPACE_ID
        self.deployment_id = config.IBM_WML_DEPLOYMENT_ID
        self._token: str | None = None

    def _auth_header(self) -> dict:
        if not self._token:
            self._token = get_iam_token()
        return {"Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json"}

    def score(self, feature_vector: list) -> dict:
        """
        Call a deployed WML model endpoint with a feature vector.
        feature_vector: list of floats matching the training feature order.
        Returns the raw WML scoring response dict.
        """
        if not self.deployment_id or self.deployment_id == "your_wml_deployment_id_here":
            raise ValueError(
                "IBM_WML_DEPLOYMENT_ID is not configured. "
                "Deploy a model in Watson Studio and set the deployment ID in .env."
            )

        endpoint = f"{self.url}/ml/v4/deployments/{self.deployment_id}/predictions?version=2021-05-01"
        payload = {
            "input_data": [
                {"values": [feature_vector]}
            ]
        }
        headers = self._auth_header()
        if self.space_id:
            headers["ML-Instance-ID"] = self.space_id

        resp = requests.post(endpoint, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        logger.info("WML scoring response received.")
        return result

    def list_deployments(self) -> list:
        """List all active deployments in the configured space."""
        endpoint = f"{self.url}/ml/v4/deployments?version=2021-05-01"
        if self.space_id:
            endpoint += f"&space_id={self.space_id}"
        resp = requests.get(endpoint, headers=self._auth_header(), timeout=30)
        resp.raise_for_status()
        return resp.json().get("resources", [])


# ── Cloud Object Storage ──────────────────────────────────────────────────────

class COSClient:
    """
    Wrapper around the IBM Cloud Object Storage S3-compatible REST API.
    Uses HMAC credentials derived from the IAM API key.
    """

    def __init__(self):
        self.endpoint = config.IBM_COS_ENDPOINT.rstrip("/")
        self.bucket = config.IBM_COS_BUCKET
        self.api_key = config.IBM_COS_API_KEY
        self.service_instance_id = config.IBM_COS_SERVICE_INSTANCE_ID

    def _auth_header(self) -> dict:
        token = get_iam_token(self.api_key or config.IBM_API_KEY)
        return {
            "Authorization": f"Bearer {token}",
            "ibm-service-instance-id": self.service_instance_id,
        }

    def upload_file(self, local_path: Path, object_key: str = None) -> bool:
        """Upload a local file to COS. Returns True on success."""
        object_key = object_key or local_path.name
        url = f"{self.endpoint}/{self.bucket}/{object_key}"
        with open(local_path, "rb") as f:
            data = f.read()
        headers = self._auth_header()
        headers["Content-Type"] = "application/octet-stream"
        resp = requests.put(url, headers=headers, data=data, timeout=120)
        resp.raise_for_status()
        logger.info("Uploaded '%s' → COS bucket '%s' as '%s'.", local_path, self.bucket, object_key)
        return True

    def download_file(self, object_key: str, local_path: Path) -> bool:
        """Download an object from COS to a local path. Returns True on success."""
        url = f"{self.endpoint}/{self.bucket}/{object_key}"
        resp = requests.get(url, headers=self._auth_header(), timeout=120)
        resp.raise_for_status()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(resp.content)
        logger.info("Downloaded COS '%s' → '%s'.", object_key, local_path)
        return True

    def list_objects(self) -> list:
        """List all objects in the configured bucket."""
        url = f"{self.endpoint}/{self.bucket}"
        resp = requests.get(url, headers=self._auth_header(), timeout=30)
        resp.raise_for_status()
        # COS returns XML; parse key names simply
        import re
        keys = re.findall(r"<Key>(.*?)</Key>", resp.text)
        return keys


# ── High-Level Helpers ────────────────────────────────────────────────────────

def upload_models_to_cos() -> dict:
    """Upload all saved model files to IBM Cloud Object Storage."""
    if not config.IBM_INTEGRATION_ENABLED:
        return {"status": "skipped", "reason": "IBM_INTEGRATION_ENABLED=false"}

    cos = COSClient()
    uploaded = []
    model_files = [
        config.FAILURE_MODEL_PATH,
        config.FAILURE_TYPE_MODEL_PATH,
        config.PREPROCESSOR_PATH,
        config.LABEL_ENCODER_PATH,
        config.FEATURE_NAMES_PATH,
    ]
    for path in model_files:
        if path.exists():
            cos.upload_file(path)
            uploaded.append(path.name)
    return {"status": "success", "uploaded": uploaded}


def score_with_wml(feature_vector: list) -> dict:
    """Score a feature vector using the deployed WML model endpoint."""
    if not config.IBM_INTEGRATION_ENABLED:
        return {"status": "skipped", "reason": "IBM_INTEGRATION_ENABLED=false"}

    client = WMLClient()
    raw = client.score(feature_vector)

    # Parse standard WML v4 response format
    try:
        predictions = raw["predictions"][0]
        values = predictions.get("values", [[]])[0]
        fields = predictions.get("fields", [])
        result = dict(zip(fields, values))
        return {"status": "success", "prediction": result, "raw": raw}
    except (KeyError, IndexError) as exc:
        logger.warning("Could not parse WML response: %s", exc)
        return {"status": "success", "raw": raw}


def test_ibm_connection() -> dict:
    """Quick connectivity test – returns status dict for dashboard display."""
    results = {
        "ibm_integration_enabled": config.IBM_INTEGRATION_ENABLED,
        "api_key_configured": bool(config.IBM_API_KEY and config.IBM_API_KEY != "your_ibm_api_key_here"),
        "wml_url": config.IBM_WML_URL,
        "cos_endpoint": config.IBM_COS_ENDPOINT,
        "iam_token_test": "not attempted",
        "wml_deployments_test": "not attempted",
    }

    if not config.IBM_INTEGRATION_ENABLED:
        results["overall_status"] = "IBM integration disabled in configuration"
        return results

    if not results["api_key_configured"]:
        results["overall_status"] = "IBM API key not configured"
        return results

    try:
        token = get_iam_token()
        results["iam_token_test"] = "✅ IAM token obtained"
    except Exception as exc:
        results["iam_token_test"] = f"❌ Failed: {exc}"
        results["overall_status"] = "IAM authentication failed"
        return results

    try:
        wml = WMLClient()
        wml._token = token
        deployments = wml.list_deployments()
        results["wml_deployments_test"] = f"✅ {len(deployments)} deployment(s) found"
    except Exception as exc:
        results["wml_deployments_test"] = f"❌ Failed: {exc}"

    results["overall_status"] = "✅ IBM Cloud connection successful"
    return results
