# webapp.py
"""
Streamlit app for Smart Crop Recommendations
- Hides Streamlit header/footer/menu (for a cleaner UI)
- Loads and trains a RandomForest (or loads pre-saved RF.pkl if present)
- Predicts recommended crop and shows message
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from PIL import Image
from sklearn import metrics
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# Helper: hide Streamlit chrome (menu/footer/header)
# -------------------------
HIDE_STREAMLIT_STYLE = """
    <style>
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    </style>
"""
st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)

# -------------------------
# Load & display top image if available
# -------------------------
if os.path.exists("crop.png"):
    try:
        img = Image.open("crop.png")
        st.image(img, use_column_width=True)
    except Exception as e:
        st.warning(f"Could not open crop.png: {e}")

# -------------------------
# Load dataset
# -------------------------
CSV_PATH = "Crop_recommendation.csv"
if not os.path.exists(CSV_PATH):
    st.error(f"Dataset file not found: {CSV_PATH}. Please ensure the CSV is present in the app folder.")
    st.stop()

df = pd.read_csv(CSV_PATH)

# Features / labels
FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
if not set(FEATURE_COLS).issubset(df.columns) or 'label' not in df.columns:
    st.error("CSV does not contain required columns. Required: " + ", ".join(FEATURE_COLS) + " and 'label'.")
    st.stop()

X = df[FEATURE_COLS]
y = df['label']

# -------------------------
# Train model if no pickle exists; otherwise load
# -------------------------
MODEL_PATH = "RF.pkl"
RF_model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            RF_model = pickle.load(f)
    except Exception as e:
        st.warning(f"Failed to load existing model '{MODEL_PATH}': {e}. Will retrain a new model.")

if RF_model is None:
    st.info("Training RandomForest model (will save to RF.pkl)...")
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)
    RF_model = RandomForestClassifier(n_estimators=20, random_state=5)
    RF_model.fit(Xtrain, Ytrain)
    preds = RF_model.predict(Xtest)
    acc = metrics.accuracy_score(Ytest, preds)
    try:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(RF_model, f)
    except Exception as e:
        st.warning(f"Could not save model to '{MODEL_PATH}': {e}")
    st.success(f"Model trained. Test accuracy: {acc:.3f}")

# -------------------------
# Prediction helper
# -------------------------
def predict_crop(n, p, k, temp, hum, ph_val, rain):
    try:
        arr = np.array([n, p, k, temp, hum, ph_val, rain]).reshape(1, -1)
        pred = RF_model.predict(arr)
        return pred[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# -------------------------
# Show crop image (optional)
# -------------------------
def show_crop_image(crop_name):
    # Expect images in 'crop_images/<crop>.jpg' or .png
    base_dir = "crop_images"
    if not os.path.isdir(base_dir):
        return
    for ext in (".jpg", ".jpeg", ".png"):
        image_path = os.path.join(base_dir, crop_name.lower() + ext)
        if os.path.exists(image_path):
            try:
                st.image(Image.open(image_path), caption=f"Recommended crop: {crop_name}", use_column_width=True)
                return
            except:
                return

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.markdown("<h1 style='text-align: center;'>SMART CROP RECOMMENDATIONS</h1>", unsafe_allow_html=True)
    st.sidebar.title("Ulavu")
    st.sidebar.header("Enter Crop Details")

    nitrogen = st.sidebar.number_input("Nitrogen (N)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
    phosphorus = st.sidebar.number_input("Phosphorus (P)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
    potassium = st.sidebar.number_input("Potassium (K)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, value=25.0, step=0.1)
    humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    ph_val = st.sidebar.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=5000.0, value=100.0, step=0.1)

    if st.sidebar.button("Predict"):
        # Basic validation: ensure not all-zero input
        inputs = np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph_val, rainfall])
        if np.isnan(inputs).any():
            st.error("Please provide valid numeric values for all fields.")
        elif (inputs == 0).all():
            st.error("All inputs are zero — please enter non-zero values to get a useful prediction.")
        else:
            crop = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph_val, rainfall)
            if crop:
                st.success(f"The recommended crop is: **{crop}**")
                show_crop_image(crop)

    st.markdown("---")
    st.write("Dataset sample:")
    st.dataframe(df.head(5))

if __name__ == "__main__":
    main()
