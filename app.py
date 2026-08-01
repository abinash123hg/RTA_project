import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Accident Severity Prediction (Keras Neural Net)",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
    .main { padding: 2rem 3rem; }
    .stButton>button {
        width: 100%;
        background-color: #2b5c8f;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover { background-color: #1d3d60; color: white; }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. DATA CLEANING & KERAS MODEL TRAINING
# ==========================================
@st.cache_resource
def train_keras_model():
    """Loads RTA Dataset.csv, cleans data, handles imbalance, and trains a Keras Neural Network."""
    dataset_path = "https://raw.githubusercontent.com/abinash123hg/rtadata/refs/heads/main/RTA%20Dataset.csv"

    try:
        if dataset_path.startswith(("http://", "https://")):
            df = pd.read_csv(dataset_path)
        else:
            if not os.path.exists(dataset_path):
                return None, None, None, None, None, f"File '{dataset_path}' not found. Please verify the local path."
            df = pd.read_csv(dataset_path)

        categorical_cols = [
            "Cause_of_accident",
            "Type_of_vehicle",
            "Area_accident_occured",
            "Lanes_or_Medians",
            "Age_band_of_driver",
            "Driving_experience",
            "Vehicle_movement",
            "Light_conditions"
        ]
        numerical_cols = [
            "Number_of_vehicles_involved",
            "Number_of_casualties"
        ]
        target_col = "Accident_severity"

        if target_col not in df.columns:
            target_col = df.columns[-1]

        # -----------------------------
        # DATA CLEANING
        # -----------------------------
        available_cat = [c for c in categorical_cols if c in df.columns]
        available_num = [c for c in numerical_cols if c in df.columns]

        df_clean = df[available_cat + available_num + [target_col]].copy()

        for col in available_cat:
            df_clean[col] = df_clean[col].fillna("Unknown").astype(str).str.strip()

        for col in available_num:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        df_clean = df_clean.dropna(subset=[target_col])

        X = df_clean[available_cat + available_num]
        y = df_clean[target_col]

        # Target Encoding
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Preprocessor with ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), available_cat),
                ('num', StandardScaler(), available_num)
            ]
        )

        X_processed = preprocessor.fit_transform(X)

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Class Weights for Imbalanced Dataset
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))

        # -----------------------------
        # TENSORFLOW / KERAS NEURAL NET
        # -----------------------------
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.4),

            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),

            Dense(32, activation='relu'),
            BatchNormalization(),

            Dense(len(classes), activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train Model
        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            class_weight=class_weight_dict,
            verbose=0
        )

        acc = model.evaluate(X_test, y_test, verbose=0)[1]

        return model, preprocessor, le, available_cat, available_num, f"Keras Model Trained Successfully (Accuracy: {acc:.2%})"

    except Exception as e:
        return None, None, None, None, None, f"Error during training: {str(e)}"

keras_model, preprocessor, le, available_cat, available_num, status_msg = train_keras_model()


# ==========================================
# 3. HEADER & INPUT FORM
# ==========================================
st.title("🧠 Accident Severity Prediction (TensorFlow / Keras)")
st.caption(status_msg)
st.markdown("---")

st.subheader("📋 Enter Accident Details")

col1, col2 = st.columns(2)

with col1:
    cause = st.selectbox("Cause of Accident", [
        "Overloading", "Changing lane to the right", "No priority to vehicle", "No priority to pedestrian",
        "Changing lane to the left", "Driving carelessly", "No distancing",
        "Overtaking", "Moving Backward", "Other", "Unknown"
    ])
    age_band = st.selectbox("Age Band of Driver", [
        "Under 18", "18-30", "31-50", "Over 51", "Unknown"
    ])
    type_vehicle = st.selectbox("Type of Vehicle", [
        "Long lorry", "Lorry (41?100Q)", "Automobile", "Public (> 45 seats)", "Lorry (11?40Q)",
        "Taxi", "Pick up upto 10Q", "Stationwagen", "Other", "Unknown"
    ])
    area = st.selectbox("Area of Accident", [
        "Outside rural areas", "Office areas", "Residential areas", "Church areas", "Industrial areas",
        "Semiurban areas", "Other", "Unknown"
    ])
    lanes = st.selectbox("Lanes or Medians", [
        "Undivided Two way", "Two-way (divided with broken lines road marking)",
        "Double carriageway (median)", "One way", "other", "Unknown"
    ])

with col2:
    light = st.selectbox("Light Conditions", [
        "Darkness - no lighting", "Daylight", "Darkness - lights lit", "Darkness - lights unlit", "Unknown"
    ])
    num_vehicles = st.number_input("Number of Vehicles Involved", min_value=1, max_value=20, value=2, step=1)
    num_casualties = st.number_input("Number of Casualties", min_value=0, max_value=50, value=1, step=1)
    driving_exp = st.selectbox("Driving Experience", [
        "No Licence", "Below 1yr", "1-2yr", "2-5yr", "5-10yr", "Above 10yr", "Unknown"
    ])
    vehicle_movement = st.selectbox("Vehicle Movement", [
        "Moving Backward", "Going straight", "Reversing", "Turnover", "Waiting to go", "Other", "Unknown"
    ])

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔮 Predict via Keras Deep Learning")


# ==========================================
# 4. PREDICTION & VISUALIZATION
# ==========================================
if predict_btn:
    st.markdown("---")
    st.subheader("🎯 Deep Learning Prediction Results")

    if keras_model is not None and preprocessor is not None and le is not None:
        inp_df = pd.DataFrame([{
            "Cause_of_accident": cause,
            "Type_of_vehicle": type_vehicle,
            "Area_accident_occured": area,
            "Lanes_or_Medians": lanes,
            "Age_band_of_driver": age_band,
            "Driving_experience": driving_exp,
            "Vehicle_movement": vehicle_movement,
            "Light_conditions": light,
            "Number_of_vehicles_involved": num_vehicles,
            "Number_of_casualties": num_casualties
        }])

        # Preprocess features properly
        X_inp = preprocessor.transform(inp_df)

        # Predict Probabilities
        proba = keras_model.predict(X_inp, verbose=0)[0]

        pred_idx = np.argmax(proba)
        pred_label = le.inverse_transform([pred_idx])[0]
        classes = le.classes_

        if "Fatal" in str(pred_label):
            st.error(f"🚨 **Predicted Severity:** {pred_label}")
        elif "Serious" in str(pred_label):
            st.warning(f"⚠️ **Predicted Severity:** {pred_label}")
        else:
            st.success(f"🟢 **Predicted Severity:** {pred_label}")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown("### Class Probabilities (Keras Output)")
            probs_df = pd.DataFrame({
                "Severity": classes,
                "Probability": proba
            }).sort_values("Probability", ascending=False).reset_index(drop=True)

            probs_df["Probability_Pct"] = (probs_df["Probability"] * 100).round(2).astype(str) + "%"
            st.table(probs_df[["Severity", "Probability_Pct"]].rename(columns={"Probability_Pct": "Probability"}))

        with res_col2:
            st.markdown("### Neural Net Probability Distribution")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')

            bars = ax.bar(
                probs_df["Severity"],
                probs_df["Probability"] * 100,
                color='#5b9bd5'
            )
            ax.set_ylabel("Probability (%)", color='white')
            ax.tick_params(colors='white')
            plt.xticks(rotation=15, color='white')
            for spine in ax.spines.values():
                spine.set_color('#333333')

            st.pyplot(fig)
    else:
        st.error("Model trained nahi ho paya hai. Path aur CSV check karein.")