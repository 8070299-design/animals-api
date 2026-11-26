import io
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI(title="Animals Classifier API (Cat / Dog / Horse)")

loaded_model = tf.saved_model.load("saved_model_animals")
infer = loaded_model.signatures["serving_default"]

CLASS_NAMES = ["cat", "dog", "horse"]

@app.get("/")
def root():
    return {"status": "ok", "message": "Use POST /predict with an image file"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((32, 32))

    img_array = np.array(image, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    input_tensor = tf.constant(img_batch)
    output_dict = infer(input_tensor)
    predictions = list(output_dict.values())[0].numpy()[0]

    predicted_idx = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(predictions[predicted_idx])

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": {
            "cat": round(float(predictions[0]), 4),
            "dog": round(float(predictions[1]), 4),
            "horse": round(float(predictions[2]), 4)
        }
    }