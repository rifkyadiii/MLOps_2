import os
import uvicorn
import tensorflow as tf
from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator

# Tentukan path model
MODEL_PATH = os.path.join('serving_model', os.listdir('serving_model')[0])

# Muat model TensorFlow
try:
    model = tf.saved_model.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Inisialisasi FastAPI app
app = FastAPI()

# Instrumentasi untuk Prometheus
Instrumentator().instrument(app).expose(app)

@app.get("/")
def read_root():
    return {"message": "Mobile Price Classification API is running!"}

@app.post("/v1/models/mobile-price-model:predict")
async def predict(request: Request):
    if not model:
        return {"error": "Model is not loaded."}, 500

    # Dapatkan serialized tf.Example dari body request
    json_body = await request.json()
    serialized_examples = [tf.constant(ex['b64'], dtype=tf.string) for ex in json_body['instances']]

    # Jalankan prediksi
    predictions = model.signatures['serving_default'](examples=tf.stack(serialized_examples))

    # Ambil nama output dari signature (biasanya 'outputs' atau nama layer terakhir)
    output_key = list(predictions.keys())[0]

    return {"predictions": predictions[output_key].numpy().tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))