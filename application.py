import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from datetime import datetime

app = Flask(__name__)

# ========================
# PATHS TO ARTIFACTS
# ========================
ARTIFACTS_BASE = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_BASE, "models", "model.pkl")
ENCODERS_PATH = os.path.join(ARTIFACTS_BASE, "processed", "feature_encoders.pkl")
TARGET_ENCODER_PATH = os.path.join(ARTIFACTS_BASE, "processed", "target_encoder.pkl")
FILL_VALUES_PATH = os.path.join(ARTIFACTS_BASE, "processed", "fill_values.pkl")

# ========================
# LOAD ALL ARTIFACTS
# ========================
try:
    model = joblib.load(MODEL_PATH)
    feature_encoders = joblib.load(ENCODERS_PATH)
    target_encoder = joblib.load(TARGET_ENCODER_PATH)
    fill_values = joblib.load(FILL_VALUES_PATH)
    
    # Get unique values for categorical dropdowns
    categorical_options = {
        col: encoder.classes_.tolist()
        for col, encoder in feature_encoders.items()
    }
    
    print(" All artifacts loaded successfully")
    print(f" Model: {type(model).__name__}")
    print(f" Feature encoders: {list(feature_encoders.keys())}")
    print(f" Target encoder classes: {target_encoder.classes_}")
    
except Exception as e:
    print(f" Error loading artifacts: {e}")
    raise

# ========================
# FEATURES DEFINITION
# ========================
NUMERICAL_FEATURES = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
    'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
    'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
    'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'
]

CATEGORICAL_FEATURES = [
    'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday'
]

# Final feature order (must match training)
FEATURE_ORDER = [
    'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
    'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
    'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
    'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
    'Temp3pm', 'RainToday', 'Year', 'Month', 'Day'
]

# ========================
# PREPROCESSING FUNCTION
# ========================
def preprocess_input(form_data):
    """
    Preprocess raw form data using saved artifacts
    
    Args:
        form_data: dict from request.form
        
    Returns:
        pd.DataFrame: Preprocessed data ready for model
    """
    try:
        # Create DataFrame from form data
        data = {}
        
        # 1. Process Date first
        date_str = form_data.get('Date', datetime.now().strftime('%Y-%m-%d'))
        date = pd.to_datetime(date_str)
        data['Year'] = date.year
        data['Month'] = date.month
        data['Day'] = date.day
        
        # 2. Process Numerical Features
        for feature in NUMERICAL_FEATURES:
            value = form_data.get(feature, '')
            
            
            if value == '' or value is None:
                # Use saved fill value (mean from training)
                data[feature] = fill_values.get(feature, 0.0)
            else:
                try:
                    data[feature] = float(value)
                except ValueError:
                    data[feature] = fill_values.get(feature, 0.0)
        
        # 3. Process Categorical Features
        for feature in CATEGORICAL_FEATURES:
            value = form_data.get(feature, '')
            
            if value == '' or value is None:
                # Use first class as default
                value = feature_encoders[feature].classes_[0]
            
            encoder = feature_encoders[feature]
            
            if value not in encoder.classes_:
                print(f" Unknown value '{value}' for {feature}, using default")
                value = encoder.classes_[0]
            
            data[feature] = encoder.transform([value])[0]
        
        df = pd.DataFrame([data], columns=FEATURE_ORDER)
        
        print(f" Preprocessed data shape: {df.shape}")
        print(f" Feature values: {df.iloc[0].to_dict()}")
        
        return df
        
    except Exception as e:
        print(f"Preprocessing error: {e}")
        raise

# ===============================================
# ROUTES
# ===============================================
@app.route("/", methods=["GET", "POST"])
def index():
    """Main page with prediction form"""
    
    prediction = None
    confidence = None
    error_message = None
    
    if request.method == "POST":
        try:
            # 1. Preprocess input data
            processed_data = preprocess_input(request.form)
            
            # 2. Make prediction (returns encoded value: 0 or 1)
            prediction_encoded = model.predict(processed_data)[0]
            
            # 3. Get prediction probability
            prediction_proba = model.predict_proba(processed_data)[0]
            confidence = max(prediction_proba) * 100
            
            # 4. Decode prediction using target encoder
            prediction_label = target_encoder.inverse_transform([prediction_encoded])[0]
            
            prediction = prediction_label
            
            print(f" Prediction: {prediction} (Confidence: {confidence:.2f}%)")
            
        except Exception as e:
            error_message = f"Prediction error: {str(e)}"
            print(f" {error_message}")
    
    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error_message,
        categorical_options=categorical_options,
        numerical_features=NUMERICAL_FEATURES
    )

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "encoders_loaded": feature_encoders is not None,
        "timestamp": datetime.now().isoformat()
    }

# ==============================
# RUN APP
# ========================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)