"""
=============================================================================
FLOOD DETECTION SYSTEM - PREDICTION ENGINE
Loads the trained Random Forest model and performs real-time risk inference.
=============================================================================
"""

import os
import json
import joblib
import pandas as pd
from utils.data_processing import (
    sanitize_input_features,
    features_to_dataframe,
    RISK_METADATA,
    LOCATIONS_DATA
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "flood_model.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

_cached_model = None
_cached_metadata = None


def load_model():
    """Load and cache the trained Scikit-Learn pipeline."""
    global _cached_model
    if _cached_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Trained model not found at: {MODEL_PATH}. Run train_model.py first.")
        _cached_model = joblib.load(MODEL_PATH)
    return _cached_model


def load_model_metadata():
    """Load model evaluation statistics and feature importances."""
    global _cached_metadata
    if _cached_metadata is None:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r") as f:
                _cached_metadata = json.load(f)
        else:
            _cached_metadata = {}
    return _cached_metadata


def predict_flood_risk(input_data, location_name="Jalandhar", data_source="User Input"):
    """
    Execute machine learning prediction for given environmental parameters.
    
    :param input_data: dict of raw or partial environmental values
    :param location_name: target Punjab city name
    :param data_source: indicator string for provenance
    :return: dict with risk level, probabilities, advisory, and input parameters
    """
    model = load_model()
    loc_meta = LOCATIONS_DATA.get(location_name, LOCATIONS_DATA["Jalandhar"])

    # Sanitize and impute parameters
    clean_features = sanitize_input_features(input_data, location_name=location_name)
    df_features = features_to_dataframe(clean_features)

    # Model inference
    pred_class = int(model.predict(df_features)[0])
    probabilities = model.predict_proba(df_features)[0]

    # Map probability percentages
    prob_dict = {
        "Low": round(float(probabilities[0]) * 100.0, 1),
        "Moderate": round(float(probabilities[1]) * 100.0, 1),
        "High": round(float(probabilities[2]) * 100.0, 1),
        "Severe": round(float(probabilities[3]) * 100.0, 1)
    }

    confidence = prob_dict[RISK_METADATA[pred_class]["label"]]
    risk_info = RISK_METADATA.get(pred_class, RISK_METADATA[0])

    response = {
        "location_name": location_name,
        "latitude": loc_meta["latitude"],
        "longitude": loc_meta["longitude"],
        "elevation_m": clean_features["elevation_m"],
        "distance_to_river_km": clean_features["distance_to_river_km"],
        "nearest_river": loc_meta["nearest_river"],
        "normal_river_level_m": loc_meta["normal_river_level_m"],
        "danger_river_level_m": loc_meta["danger_river_level_m"],
        "inputs": clean_features,
        "prediction": {
            "risk_level": pred_class,
            "risk_label": risk_info["label"],
            "color": risk_info["color"],
            "badge_class": risk_info["badge_class"],
            "confidence_pct": confidence,
            "probabilities": prob_dict,
            "summary": risk_info["summary"],
            "advisory": risk_info["advisory"],
            "action": risk_info["action"]
        },
        "data_source": data_source
    }

    return response
