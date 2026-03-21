# Damage Severity and Repair Cost Intelligence

This project is a Flask-based web application for analyzing damage photos of phones and laptops, estimating repair cost in INR, and giving repair-worth recommendations with reasons.

## What The App Does

1. Accepts only phone/laptop images (strict validation).
2. Predicts severity: Minor, Moderate, Severe.
3. Generates a Grad-CAM visual explanation.
4. Estimates repair cost (fixed INR output).
5. Compares optional shop quote and returns Yes/No decision with reason.
6. Provides a downloadable PDF report with all prediction details.

## Main Features

- Two image source options on first page:
   - Take Photo (camera capture)
   - Upload Photo (file picker)
- Device type selection: Phone or Laptop
- Optional shop quote input:
   - Quoted amount
   - Quoted currency
   - Shop details/notes
- Strict non-device rejection:
   - Unrelated images are blocked with HTTP 400 and a clear reason
- INR cost display
- Repair-worth guidance:
   - AI recommendation (worth / not worth) + reason
   - Shop quote assessment (yes/no) + reason
- PDF export from prediction page

## Tech Stack

- Backend: Flask
- ML: TensorFlow / Keras (EfficientNet + MobileNetV2 validator)
- Imaging: Pillow
- PDF: ReportLab
- Frontend: HTML/CSS/JavaScript

## Project Structure

- main.py: Flask app entry point
- api/routes.py: Prediction API, quote assessment, PDF report endpoint
- models/classification_model.py: Damage classifier loading and inference
- models/cost_estimation_model.py: Device-aware cost estimation
- models/train_classifier.py: Training script for classifier weights
- models/weights/: Saved trained model weights
- utils/device_validator.py: Strict phone/laptop image validation
- utils/gradcam.py: Grad-CAM overlay generation
- utils/currency.py: Currency conversion helpers
- frontend/index.html: Input page
- frontend/prediction.html: Result page
- frontend/upload.js: Input-page logic
- frontend/app.js: Prediction-page logic and PDF download

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run The Server

```bash
python main.py
```

Open:

- http://127.0.0.1:8000

## API Endpoints

- POST /api/predict
   - Input (multipart/form-data):
      - image (required)
      - device_type (required: phone | laptop)
      - region_code (optional)
      - shop_quote_amount (optional)
      - shop_quote_currency (optional)
      - shop_quote_details (optional)
   - Output:
      - severity, probabilities, confidence
      - gradcam_base64
      - cost_estimate (INR)
      - repair_recommendation (decision + reason)
      - shop_quote_assessment (if quote provided)

- POST /api/predict/report
   - Same input as /api/predict
   - Output: downloadable PDF report containing website name and all prediction details

## Training The Classifier

Run:

```bash
python models/train_classifier.py
```

Expected training data folders:

- data/Image_brokenphones
- data/Image_phones

Output weights file:

- models/weights/efficientnetb0.weights.h5

## Important Notes

- The app enforces device-image validation before prediction; unrelated photos are rejected.
- Cost output is currently forced to INR.
- Shop quote decision is an assistive recommendation, not a legal/financial guarantee.
- For best production accuracy, train with a larger and balanced dataset across real damage classes.
