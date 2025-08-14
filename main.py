import os
import base64
import logging
from typing import Any, Dict

import numpy as np
import requests
import tensorflow as tf
from fastapi import FastAPI, Request, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mobile-price-api")

# Config via env
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/serving_model")
MODEL_URL = os.environ.get("MODEL_URL")  # optional: download model at startup
PORT = int(os.environ.get("PORT", 8080))

app = FastAPI(title="Mobile Price Classification API")
Instrumentator().instrument(app).expose(app)  # /metrics

def download_and_extract_model(model_url: str, dest_dir: str) -> None:
    """Optional: download and extract zip model artifact from MODEL_URL."""
    logger.info("Downloading model from %s", model_url)
    r = requests.get(model_url, stream=True, timeout=120)
    r.raise_for_status()
    tmp_path = "/tmp/model_download.zip"
    with open(tmp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    import zipfile
    with zipfile.ZipFile(tmp_path, "r") as z:
        z.extractall(dest_dir)
    logger.info("Model downloaded/extracted to %s", dest_dir)

def find_model_path(model_parent_dir: str) -> str:
    """Return first subdirectory as SavedModel path."""
    if not os.path.isdir(model_parent_dir):
        raise FileNotFoundError(f"Model directory not found: {model_parent_dir}")
    entries = sorted([p for p in os.listdir(model_parent_dir) if os.path.isdir(os.path.join(model_parent_dir, p))])
    if not entries:
        raise FileNotFoundError(f"No subfolder (saved model) found in: {model_parent_dir}")
    return os.path.join(model_parent_dir, entries[0])

# Load model (optional: download first)
model = None
try:
    if MODEL_URL and (not os.path.exists(MODEL_DIR) or len(os.listdir(MODEL_DIR)) == 0):
        os.makedirs(MODEL_DIR, exist_ok=True)
        download_and_extract_model(MODEL_URL, MODEL_DIR)

    model_path = find_model_path(MODEL_DIR)
    logger.info("Loading TF SavedModel from %s", model_path)
    model = tf.saved_model.load(model_path)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.exception("Failed to load model: %s", e)
    model = None

@app.get("/", tags=["health"])
def read_root():
    return {"message": "Mobile Price Classification API is running!", "model_loaded": model is not None}

def _serialize_predictions(predictions: Dict[str, Any]) -> Dict[str, Any]:
    def convert(obj):
        if isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj
    return {k: convert(v) for k, v in predictions.items()}

@app.post("/v1/models/mobile-price-model:predict", tags=["prediction"])
async def predict(request: Request):
    """
    Supports:
    A) tf.Example base64 JSON: {"instances":[{"b64":"..."}]}
    B) Raw numeric: {"instances":[[f1,f2,...],[...]]}
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    payload = await request.json()
    instances = payload.get("instances") or payload.get("inputs") or payload.get("data")
    if instances is None:
        raise HTTPException(status_code=400, detail="Missing 'instances' in request body.")

    try:
        signature = model.signatures.get("serving_default") or list(model.signatures.values())[0]
    except Exception:
        signature = None
    if signature is None:
        raise HTTPException(status_code=500, detail="Model has no callable signatures.")

    # Try to infer input name
    try:
        sig_input_map = signature.structured_input_signature[1]
        input_names = list(sig_input_map.keys())
    except Exception:
        input_names = []
    input_key = input_names[0] if input_names else None

    # Case A: base64 tf.Example
    if isinstance(instances, list) and len(instances) > 0 and isinstance(instances[0], dict) and "b64" in instances[0]:
        try:
            decoded = [base64.b64decode(item["b64"]) for item in instances]
            tensor = tf.constant(decoded, dtype=tf.string)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode base64: {e}")
        raw_preds = signature(**{input_key: tensor}) if input_key else signature(tensor)
    else:
        # Case B: numeric
        try:
            arr = np.array(instances)
            tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to convert instances to tensor: {e}")
        raw_preds = signature(**{input_key: tensor}) if input_key else signature(tensor)

    preds = _serialize_predictions(raw_preds)
    return {"predictions": preds}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")
