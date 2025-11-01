# Handwritten Digit Recognition Web App

A modern web application for recognizing handwritten digits (0-9) using a trained CNN model on the MNIST dataset. Built with FastAPI and a beautiful, responsive UI.

## Features

- **Interactive Drawing Canvas**: Draw digits directly in the browser
- **Image Upload**: Upload images of handwritten digits
- **Real-time Predictions**: Get instant predictions with confidence scores
- **Visual Feedback**: See all probability distributions for digits 0-9
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Beautiful UI/UX**: Modern gradient design with smooth animations

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the FastAPI server:

```bash
python main.py
```

Or use uvicorn directly:

```bash
uvicorn main:app --reload
```

2. Open your web browser and navigate to:

```
http://localhost:8000
```

## How to Use

### Method 1: Draw a Digit

1. Use your mouse or finger (on touch devices) to draw a digit on the canvas
2. Click "Predict Drawing" to get the prediction
3. View the results with confidence scores

### Method 2: Upload an Image

1. Click the upload area or drag and drop an image
2. Preview your original and processed (28x28) images
3. Click "Predict Digit" to get the prediction
4. View detailed probability distribution for all digits

## API Endpoints

### GET `/`
Returns the main HTML interface

### POST `/predict`
Accepts an image file and returns prediction results

**Response:**
```json
{
  "success": true,
  "predicted_digit": 7,
  "confidence": 0.9856,
  "all_probabilities": {
    "0": 0.0001,
    "1": 0.0002,
    ...
    "7": 0.9856,
    ...
  }
}
```

### GET `/health`
Health check endpoint to verify the model is loaded

## Project Structure

```
character-recog/
├── model/
│   └── my_cnn_model.keras    # Trained MNIST model
├── templates/
│   └── index.html            # Web UI
├── main.py                   # FastAPI application
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Technical Details

- **Backend**: FastAPI (Python)
- **Frontend**: Pure HTML, CSS, JavaScript
- **Model**: Keras/TensorFlow CNN trained on MNIST
- **Image Processing**: PIL (Pillow)
- **Input Size**: 28x28 grayscale images
- **Output**: 10 classes (digits 0-9)

## Image Processing Pipeline

1. Convert uploaded image to grayscale
2. Resize to 28x28 pixels (MNIST standard)
3. Invert colors if needed (MNIST expects white digits on black background)
4. Normalize pixel values to [0, 1]
5. Reshape for model input
6. Generate predictions

## Tips for Best Results

- Draw digits clearly and centered
- Use dark pen/marker on white paper when uploading photos
- Ensure good contrast in uploaded images
- Draw digits similar to how you would write them naturally

## Troubleshooting

**Model not loading?**
- Ensure `model/my_cnn_model.keras` exists
- Check that TensorFlow is properly installed

**Low prediction accuracy?**
- Ensure digits are centered and clear
- Try inverting colors if using light digits on dark background
- Make sure the digit fills most of the space

## License

This project is open source and available for educational purposes.
