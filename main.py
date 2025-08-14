import os
import base64
import logging
from typing import Any, Dict

import numpy as np
import requests
import tensorflow as tf
from fastapi import FastAPI, Request, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

# Setup logging dasar
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mobile-price-api")

# Konfigurasi dari environment variable
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/serving_model")
PORT = int(os.environ.get("PORT", 7860))

# Inisialisasi FastAPI dan Prometheus
app = FastAPI(title="Mobile Price Classification API")
Instrumentator().instrument(app).expose(app)

# Fungsi untuk menemukan path model di dalam folder serving
def find_model_path(model_parent_dir: str) -> str:
    if not os.path.isdir(model_parent_dir):
        raise FileNotFoundError(f"Model directory not found: {model_parent_dir}")
    entries = sorted([p for p in os.listdir(model_parent_dir) if os.path.isdir(os.path.join(model_parent_dir, p))])
    if not entries:
        raise FileNotFoundError(f"No subfolder (saved model) found in: {model_parent_dir}")
    return os.path.join(model_parent_dir, entries[0])

# Memuat model saat aplikasi dimulai
model = None
try:
    model_path = find_model_path(MODEL_DIR)
    logger.info("Loading TF SavedModel from %s", model_path)
    model = tf.saved_model.load(model_path)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.exception("Failed to load model: %s", e)
    model = None

# Endpoint untuk health check
@app.get("/", tags=["Health Check"])
def read_root():
    return {"message": "Mobile Price Classification API is running!", "model_loaded": model is not None}

# Fungsi untuk serialisasi output prediksi
def _serialize_predictions(predictions: Dict[str, Any]) -> Dict[str, Any]:
    def convert(obj):
        if isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
        return obj
    return {k: convert(v) for k, v in predictions.items()}

# Endpoint untuk prediksi
@app.post("/v1/models/mobile-price-model:predict", tags=["Prediction"])
async def predict(request: Request):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or unavailable.")

    payload = await request.json()
    instances = payload.get("instances")
    if not instances:
        raise HTTPException(status_code=400, detail="Missing 'instances' key in request body.")

    try:
        # Mendukung format tf.Example yang di-encode base64
        if isinstance(instances[0], dict) and "b64" in instances[0]:
            decoded = [base64.b64decode(item["b64"]) for item in instances]
            tensor = tf.constant(decoded, dtype=tf.string)
        else:
            # Mendukung format list numerik biasa
            tensor = tf.convert_to_tensor(instances, dtype=tf.float32)
        
        signature = model.signatures["serving_default"]
        raw_preds = signature(examples=tensor) # Menggunakan nama input 'examples'
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process request: {e}")

    preds = _serialize_predictions(raw_preds)
    return {"predictions": preds[list(preds.keys())[0]]}