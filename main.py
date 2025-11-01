from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI(title="MNIST Digit Recognition API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
MODEL_PATH = "model/my_cnn_model.keras"
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
        print(f"Model input shape: {model.input_shape}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the uploaded image for MNIST model prediction.
    """
    # Convert to grayscale
    image = image.convert('L')

    # Resize to 28x28 (MNIST standard size)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to numpy array
    img_array = np.array(image)

    # Invert colors if needed (MNIST has white digits on black background)
    # Check if the image has dark digits on light background
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0

    # Reshape for model input
    # Check model input shape and adjust accordingly
    if len(model.input_shape) == 4:
        if model.input_shape[-1] == 1:
            # (batch, height, width, channels)
            img_array = img_array.reshape(1, 28, 28, 1)
        else:
            # (batch, height, width, channels) - might need 3 channels
            img_array = img_array.reshape(1, 28, 28, 1)
    else:
        # Flatten if needed
        img_array = img_array.reshape(1, 784)

    return img_array

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Serve the main HTML page.
    """
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict the digit from an uploaded image.
    """
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Preprocess image
        processed_image = preprocess_image(image)

        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_digit])

        # Get all probabilities
        all_probabilities = {
            str(i): float(predictions[0][i])
            for i in range(10)
        }

        return JSONResponse(content={
            "success": True,
            "predicted_digit": predicted_digit,
            "confidence": confidence,
            "all_probabilities": all_probabilities
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
