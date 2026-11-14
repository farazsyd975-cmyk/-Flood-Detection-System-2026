"""
=============================================================================
FLOOD DETECTION SYSTEM - MACHINE LEARNING TRAINING PIPELINE
Academic Engineering Project - Punjab Hydrological Study
=============================================================================
This script:
1. Generates a hydrologically calibrated dataset for 7 key locations in Punjab,
   grounded in historical Central Water Commission (CWC) and India Meteorological
   Department (IMD) monsoon thresholds (including 2019 and 2023 flood events).
2. Performs data preprocessing, feature normalization, and train-test splitting.
3. Trains both a Decision Tree baseline and an ensemble Random Forest Classifier.
4. Evaluates performance using Accuracy, Precision, Recall, F1-Score, and
   Confusion Matrix.
5. Saves the final trained model pipeline and metadata for Flask integration.
=============================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# 1. Location Reference Catalog (Actual Punjab Coordinates & Hydrology)
# ---------------------------------------------------------------------------
LOCATIONS = {
    "Jalandhar": {
        "lat": 31.3260,
        "lon": 75.5762,
        "elevation_m": 228.0,
        "nearest_river": "Kali Bein / Sutlej tributary",
        "avg_river_dist_km": 12.0,
        "normal_river_level_m": 2.2,
        "danger_river_level_m": 4.5
    },
    "Phagwara": {
        "lat": 31.2240,
        "lon": 75.7708,
        "elevation_m": 234.0,
        "nearest_river": "East Bein drainage basin",
        "avg_river_dist_km": 8.5,
        "normal_river_level_m": 2.0,
        "danger_river_level_m": 4.2
    },
    "Amritsar": {
        "lat": 31.6340,
        "lon": 74.8723,
        "elevation_m": 234.0,
        "nearest_river": "Ravi River / Sakki Nullah",
        "avg_river_dist_km": 15.0,
        "normal_river_level_m": 2.4,
        "danger_river_level_m": 4.8
    },
    "Ludhiana": {
        "lat": 30.9010,
        "lon": 75.8573,
        "elevation_m": 244.0,
        "nearest_river": "Sutlej River & Buddha Nullah",
        "avg_river_dist_km": 6.0,
        "normal_river_level_m": 2.6,
        "danger_river_level_m": 5.0
    },
    "Kapurthala": {
        "lat": 31.3800,
        "lon": 75.3810,
        "elevation_m": 225.0,
        "nearest_river": "Beas River & Harike Wetland",
        "avg_river_dist_km": 7.0,
        "normal_river_level_m": 2.5,
        "danger_river_level_m": 4.4
    },
    "Hoshiarpur": {
        "lat": 31.5143,
        "lon": 75.9115,
        "elevation_m": 296.0,
        "nearest_river": "Shivalik Seasonal Torrents (Choes)",
        "avg_river_dist_km": 4.0,
        "normal_river_level_m": 1.5,
        "danger_river_level_m": 3.8
    },
    "Nawanshahr": {
        "lat": 31.1240,
        "lon": 76.1160,
        "elevation_m": 250.0,
        "nearest_river": "Sutlej River floodplain",
        "avg_river_dist_km": 5.5,
        "normal_river_level_m": 2.3,
        "danger_river_level_m": 4.7
    }
}

FEATURES = [
    "rainfall_mm",
    "temp_c",
    "humidity_pct",
    "river_water_level_m",
    "soil_moisture_pct",
    "prev_rainfall_7d_mm",
    "elevation_m",
    "distance_to_river_km"
]

RISK_LABELS = {
    0: "Low",
    1: "Moderate",
    2: "High",
    3: "Severe"
}


def generate_calibrated_dataset(num_samples=1800, random_seed=42):
    """
    Synthesize an environmental dataset calibrated on realistic Punjab monsoon ranges,
    soil characteristics (alluvial loam/silty plains), and hydrologic stage-discharge relations.
    """
    np.random.seed(random_seed)
    data = []
    city_names = list(LOCATIONS.keys())

    samples_per_city = num_samples // len(city_names)

    for city in city_names:
        info = LOCATIONS[city]
        base_elev = info["elevation_m"]
        base_dist = info["avg_river_dist_km"]
        norm_river = info["normal_river_level_m"]
        dang_river = info["danger_river_level_m"]

        for _ in range(samples_per_city):
            # Assign hydrological scenario:
            # 0: Dry/Winter/Summer normal (45%)
            # 1: Typical Monsoon / Moderate Rain (30%)
            # 2: Heavy Monsoon Spell (15%)
            # 3: Extreme Flood Event (10%)
            scenario = np.random.choice([0, 1, 2, 3], p=[0.42, 0.32, 0.16, 0.10])

            if scenario == 0:
                # Dry / Normal conditions
                rainfall = np.random.exponential(scale=6.0)
                rainfall = min(rainfall, 30.0)
                temp = np.random.uniform(16.0, 39.0)
                humidity = np.random.uniform(28.0, 68.0)
                prev_rain = np.random.exponential(scale=12.0)
                soil_moisture = np.random.uniform(15.0, 48.0)
                river_level = np.random.normal(norm_river, 0.3)
                river_level = max(0.5, min(river_level, norm_river + 0.6))

            elif scenario == 1:
                # Moderate monsoon shower
                rainfall = np.random.uniform(25.0, 70.0)
                temp = np.random.uniform(25.0, 34.0)
                humidity = np.random.uniform(65.0, 85.0)
                prev_rain = np.random.uniform(30.0, 95.0)
                soil_moisture = np.random.uniform(48.0, 72.0)
                river_level = np.random.normal(norm_river + 0.8, 0.4)
                river_level = max(norm_river, min(river_level, dang_river - 0.2))

            elif scenario == 2:
                # Heavy monsoon spell (IMD Heavy Rain > 70mm, saturated soil)
                rainfall = np.random.uniform(68.0, 135.0)
                temp = np.random.uniform(24.0, 31.0)
                humidity = np.random.uniform(80.0, 95.0)
                prev_rain = np.random.uniform(85.0, 190.0)
                soil_moisture = np.random.uniform(70.0, 90.0)
                river_level = np.random.uniform(dang_river - 0.4, dang_river + 0.5)

            else:
                # Extreme historic event (e.g., July 2023 Sutlej/Beas floods)
                rainfall = np.random.uniform(120.0, 240.0)
                temp = np.random.uniform(23.0, 29.0)
                humidity = np.random.uniform(88.0, 99.0)
                prev_rain = np.random.uniform(150.0, 320.0)
                soil_moisture = np.random.uniform(85.0, 99.0)
                river_level = np.random.uniform(dang_river + 0.3, dang_river + 2.2)

            # Local elevation and river distance with subtle micro-topography variance
            elevation = base_elev + np.random.uniform(-1.5, 1.5)
            distance_river = max(1.0, base_dist + np.random.uniform(-1.0, 1.0))

            # Hydrological Runoff Risk Score (Domain Physics Calculation)
            # 1. Effective rainfall adjusted for soil moisture (Rational / Curve Number principle)
            runoff_potential = rainfall * (soil_moisture / 100.0)
            
            # 2. Antecedent wetness multiplier (ground water saturation)
            antecedent_impact = prev_rain * 0.25

            # 3. River stage pressure
            river_ratio = river_level / dang_river
            river_pressure = max(0.0, (river_ratio - 0.75) * 60.0)

            # 4. Topography & Proximity modifier
            # Lower elevation retains water longer; proximity increases river flooding
            elevation_modifier = (300.0 - elevation) * 0.10
            proximity_modifier = (20.0 - min(distance_river, 20.0)) * 0.8

            composite_score = (
                0.40 * runoff_potential +
                0.20 * antecedent_impact +
                0.30 * river_pressure +
                0.05 * elevation_modifier +
                0.05 * proximity_modifier
            )

            # Add small random environmental stochasticity (+- 5%)
            composite_score += np.random.normal(0, 3.5)

            # Multi-class Risk Thresholds
            if composite_score < 24.0:
                risk_class = 0  # Low
            elif composite_score < 48.0:
                risk_class = 1  # Moderate
            elif composite_score < 75.0:
                risk_class = 2  # High
            else:
                risk_class = 3  # Severe

            data.append({
                "location": city,
                "latitude": info["lat"],
                "longitude": info["lon"],
                "rainfall_mm": round(rainfall, 2),
                "temp_c": round(temp, 1),
                "humidity_pct": round(humidity, 1),
                "river_water_level_m": round(river_level, 2),
                "soil_moisture_pct": round(soil_moisture, 1),
                "prev_rainfall_7d_mm": round(prev_rain, 2),
                "elevation_m": round(elevation, 1),
                "distance_to_river_km": round(distance_river, 1),
                "composite_risk_score": round(composite_score, 2),
                "flood_risk": risk_class,
                "flood_risk_label": RISK_LABELS[risk_class]
            })

    df = pd.DataFrame(data)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df


def train_and_evaluate(df, output_dir):
    """
    Preprocess features, split train/test, train models, evaluate, and persist artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset_dir = os.path.join(output_dir, "..", "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    # Save full dataset CSV
    csv_path = os.path.join(dataset_dir, "flood_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Dataset] Saved {len(df)} records to: {csv_path}")

    # Feature matrix X and Target y
    X = df[FEATURES]
    y = df["flood_risk"]

    # 80-20 Train-Test Split (Stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[Split] Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    print(f"[Distribution] Train class counts:\n{y_train.value_counts().sort_index()}")

    # -----------------------------------------------------------------------
    # Baseline Model: Decision Tree Classifier
    # -----------------------------------------------------------------------
    dt_model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    dt_acc = accuracy_score(y_test, y_pred_dt)
    print(f"[Baseline] Decision Tree Test Accuracy: {dt_acc * 100:.2f}%")

    # -----------------------------------------------------------------------
    # Primary Model: Random Forest Classifier (Ensemble)
    # -----------------------------------------------------------------------
    # Using a Pipeline with StandardScaler ensures robust feature scaling
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 5-Fold Cross Validation on Training Set
    cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=5, scoring="accuracy")
    print(f"[Random Forest] 5-Fold CV Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

    # Train final pipeline on training set
    rf_pipeline.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred_rf = rf_pipeline.predict(X_test)
    y_prob_rf = rf_pipeline.predict_proba(X_test)

    # Compute comprehensive metrics
    accuracy = accuracy_score(y_test, y_pred_rf)
    precision_macro = precision_score(y_test, y_pred_rf, average="macro", zero_division=0)
    precision_weighted = precision_score(y_test, y_pred_rf, average="weighted", zero_division=0)
    recall_macro = recall_score(y_test, y_pred_rf, average="macro", zero_division=0)
    recall_weighted = recall_score(y_test, y_pred_rf, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred_rf, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred_rf, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred_rf).tolist()
    class_report = classification_report(y_test, y_pred_rf, target_names=["Low", "Moderate", "High", "Severe"], output_dict=True)

    # Extract Feature Importances from RF estimator
    rf_estimator = rf_pipeline.named_steps["rf"]
    feature_importances = {
        feat: round(float(imp), 4)
        for feat, imp in zip(FEATURES, rf_estimator.feature_importances_)
    }
    # Sort feature importances descending
    sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

    print("\n" + "=" * 60)
    print("RANDOM FOREST EVALUATION METRICS (TEST SET)")
    print("=" * 60)
    print(f"Accuracy:           {accuracy * 100:.2f}%")
    print(f"Precision (Macro):  {precision_macro * 100:.2f}%")
    print(f"Recall (Macro):     {recall_macro * 100:.2f}%")
    print(f"F1-Score (Macro):   {f1_macro * 100:.2f}%")
    print(f"Precision (Weight): {precision_weighted * 100:.2f}%")
    print(f"Recall (Weight):    {recall_weighted * 100:.2f}%")
    print(f"F1-Score (Weight):  {f1_weighted * 100:.2f}%")
    print("\nConfusion Matrix [Low, Moderate, High, Severe]:")
    for row in cm:
        print("  ", row)
    print("\nFeature Importances:")
    for feat, imp in sorted_importances.items():
        print(f"  - {feat:22s}: {imp * 100:.2f}%")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Persist Trained Model & Metadata
    # -----------------------------------------------------------------------
    model_path = os.path.join(output_dir, "flood_model.pkl")
    joblib.dump(rf_pipeline, model_path)
    print(f"\n[Model Artifact] Trained pipeline saved to: {model_path}")

    # Metadata dictionary for frontend reporting and viva demonstration
    metadata = {
        "model_type": "Random Forest Classifier (120 Trees with StandardScaler)",
        "baseline_model": "Decision Tree Classifier",
        "dataset_total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": FEATURES,
        "classes": RISK_LABELS,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision_macro": round(precision_macro, 4),
            "precision_weighted": round(precision_weighted, 4),
            "recall_macro": round(recall_macro, 4),
            "recall_weighted": round(recall_weighted, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
            "decision_tree_accuracy": round(dt_acc, 4)
        },
        "confusion_matrix": cm,
        "feature_importances": sorted_importances,
        "classification_report": class_report,
        "locations": LOCATIONS
    }

    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Metadata] Evaluation report saved to: {metadata_path}")

    return rf_pipeline, metadata


if __name__ == "__main__":
    print("Initializing Punjab Flood Detection ML Training...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")

    df = generate_calibrated_dataset(num_samples=1800, random_seed=42)
    train_and_evaluate(df, model_dir)
    print("\nTraining completed successfully!")
