# FLOOD DETECTION SYSTEM
### An Explainable Machine Learning & Geospatial Risk Assessment Platform for Punjab River Basins

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Flask](https://img.shields.io/badge/Framework-Flask%203.1-black.svg)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%20Random%20Forest-orange.svg)
![Leaflet](https://img.shields.io/badge/Maps-Leaflet.js%20%7C%20OSM-brightgreen.svg)
![Database](https://img.shields.io/badge/Database-SQLite%203-blueviolet.svg)
![Deploy](https://img.shields.io/badge/Deploy-Render-46e3b7.svg)

---

## 1. Project Overview

The **Flood Detection System** is an engineering AI/ML application engineered to evaluate, classify, and visualize early flood hazard risks across seven vulnerable districts in **Punjab, India**:
1. **Jalandhar** (Doaba plain / Kali Bein tributary)
2. **Phagwara** (East Bein drainage basin)
3. **Amritsar** (Ravi River basin & Sakki Nullah)
4. **Ludhiana** (Sutlej River & Buddha Nullah channel)
5. **Kapurthala** (Beas River & Harike Wetland confluence / Mand area)
6. **Hoshiarpur** (Shivalik foothill seasonal torrents / *choes*)
7. **Nawanshahr (SBS Nagar)** (Sutlej River floodplain below Ropar barrage)

The system bridges atmospheric telemetry, soil hydrologic physics, user authentication, and geospatial cartography to provide real-time probabilistic risk ratings: **Low Risk (Green)**, **Moderate Risk (Yellow)**, **High Risk (Orange)**, and **Severe Risk (Red)**.

---

## 2. Key Engineering Highlights

- **User Authentication**: Secure SQLite-backed email login system with password hashing (`werkzeug.security`) and session state management.
- **Genuine Local Machine Learning**: Core prediction engine uses a locally trained **Random Forest Classifier (120 Estimators with StandardScaler)** saved as a `.pkl` pipeline. It generates predictions with full class probability breakdowns offline or online.
- **Fail-Safe Offline Mode**: If external weather APIs are disconnected or blocked, the system automatically falls back to verified regional seasonal baselines for that exact Punjab district, clearly labeling the data provenance.
- **Interactive OpenStreetMap Visualization**: Uses Leaflet.js to render authentic Punjab geography with dynamic risk-coded circle markers, interactive popups, coordinate telemetry, and fast zoom transitions.
- **"What-If" Simulation Mode**: Includes interactive sliders and preset scenario buttons (*Normal Dry Day*, *Active Monsoon Shower*, *Heavy Inundation Spell*, *July 2023 Historic Flood*) to test model sensitivities live.
- **Persistent SQLite Audit Trail**: Every automated prediction or manual simulation is recorded with full environmental inputs, confidence scores, and timestamps in `database/flood.db`.
- **Clean University-Grade Codebase**: Structured according to standard Flask and Python design patterns, completely free of generic AI dashboard bloat, fake counters, or non-functional placeholder buttons.

---

## 3. User Authentication & Default Login

The platform features a simple, clean email login and registration interface (`/login`, `/register`).

| Email Address | Default Password | Description |
|---|---|---|
| `user@floodsystem.com` | `password123` | Default system user account |

*(Users can also register a new account with their own email address via `/register`)*

---

## 4. Machine Learning Architecture & Evaluation

### 4.1 Feature Schema & Hydrological Physics
The model evaluates 8 domain-grounded hydrological features:
1. `rainfall_mm`: 24-hour cumulative rainfall (mm).
2. `temp_c`: Ambient surface temperature (°C).
3. `humidity_pct`: Relative atmospheric humidity (%).
4. `river_water_level_m`: Water gauge level of nearest river/drainage relative to local zero datum (m).
5. `soil_moisture_pct`: Volumetric moisture saturation of upper soil column (%).
6. `prev_rainfall_7d_mm`: 7-day antecedent rainfall (mm) representing ground pre-saturation.
7. `elevation_m`: Elevation above Mean Sea Level (MSL) in meters.
8. `distance_to_river_km`: Proximity from city center to primary natural drainage channel (km).

### 4.2 Model Evaluation Metrics (Held-Out Test Set)
Evaluated on an 80/20 stratified split (1,439 training samples, 360 held-out test samples):

| Metric | Decision Tree (Baseline) | Random Forest (Primary Pipeline) |
|---|---|---|
| **Test Accuracy** | 93.33% | **95.83%** |
| **5-Fold Cross-Validation Accuracy** | 89.40% | **94.72% (±1.78%)** |
| **Precision (Weighted)** | 92.95% | **95.72%** |
| **Recall (Weighted)** | 93.33% | **95.83%** |
| **F1-Score (Weighted)** | 93.02% | **95.76%** |
| **Macro F1-Score** | 83.30% | **88.71%** |

### 4.3 Confusion Matrix (360 Test Samples)
```
                  PREDICTED CLASS
             Low   Moderate   High   Severe
ACTUAL Low   265      0         0       0
      Mod      4     37         1       0
     High      0      4        19       2
   Severe      0      0         4      24
```
- **Zero False Negatives for Severe Inundation**: No high/severe event was ever misclassified as Low, satisfying safety requirements.
- **100% Specificity on Normal Conditions**: 265/265 baseline dry records correctly classified as Low Risk.

### 4.4 Gini Feature Importance Ranking
1. **24h Rainfall (`rainfall_mm`)**: 32.91%
2. **7-Day Antecedent Rainfall (`prev_rainfall_7d_mm`)**: 23.91%
3. **Soil Moisture Saturation (`soil_moisture_pct`)**: 15.96%
4. **River Water Level (`river_water_level_m`)**: 11.67%
5. **Relative Humidity (`humidity_pct`)**: 10.46%
6. **Ambient Temperature (`temp_c`)**: 2.13%
7. **Elevation (`elevation_m`)**: 1.54%
8. **River Proximity (`distance_to_river_km`)**: 1.42%

---

## 5. Project Directory Structure

```
Flood-Detection-System/
│
├── app.py                      # Flask main server, authentication & API routes
├── train_model.py              # ML training script, evaluation & pipeline persistence
├── requirements.txt            # Python dependencies (pinned versions)
├── Procfile                    # Render/Gunicorn startup command
├── render.yaml                 # Infrastructure as Code configuration for Render
├── README.md                   # Full documentation & setup guide
├── .gitignore                  # Clean repository exclusions
│
├── model/
│   ├── flood_model.pkl         # Trained Random Forest Scikit-Learn pipeline
│   └── model_metadata.json     # Comprehensive evaluation metrics & feature importances
│
├── dataset/
│   ├── flood_data.csv          # Hydrologically calibrated dataset (1,799 rows)
│   └── dataset_metadata.md     # Feature data dictionary & physical formulas
│
├── templates/
│   ├── base.html               # Master layout with navbar & user badge
│   ├── index.html              # Main Punjab interactive map & risk predictor
│   ├── model_info.html         # Model metrics, confusion matrix & technical FAQ
│   ├── history.html            # SQLite historical audit log table
│   ├── login.html              # User email login page
│   └── register.html           # New user registration page
│
├── static/
│   ├── css/
│   │   └── style.css           # Clean university-grade design (navy/white theme)
│   ├── js/
│   │   ├── map.js              # Leaflet.js map, dynamic pin pulse markers & popups
│   │   └── main.js             # Telemetry fetching, slider bindings & prediction controller
│   └── images/
│
├── database/
│   ├── db.py                   # SQLite connection helpers, query routines & auth
│   └── flood.db                # SQLite embedded database file (auto-initialized)
│
├── utils/
│   ├── data_processing.py      # Location definitions, input validation & schemas
│   ├── weather_service.py      # Open-Meteo live observation service with baseline fallback
│   └── prediction.py           # Pipeline loading & real-time probabilistic inference
│
└── tests/
    └── test_system.py          # Automated test suite (13 unit & integration tests)
```

---

## 6. Local Setup & Execution Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.
- Internet connection (optional; required only for initial package install and live Open-Meteo queries).

### Step 1: Clone or Extract the Project
Navigate to the project root directory:
```bash
cd Flood-Detection-System
```

### Step 2: Create and Activate a Virtual Environment
**On Windows (PowerShell / Command Prompt):**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Automated Test Suite
Verify all 13 unit, integration, and authentication tests:
```bash
python tests/test_system.py
```
Expected output:
```
Ran 13 tests in 9.363s
OK
```

### Step 5: Start the Flask Web Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```
To log in, click **Login** in the top navigation bar or go to:
```
http://127.0.0.1:5000/login
```

---

## 7. Deployment on Render

This application is configured for deployment on **Render** using Gunicorn.

### Deployment Steps:
1. Push the `Flood-Detection-System` repository to GitHub or GitLab.
2. Log in to [Render](https://render.com) and click **New +** ➔ **Web Service**.
3. Select your repository.
4. Configure the service settings:
   - **Name**: `punjab-flood-detection-system`
   - **Region**: Singapore or Frankfurt (choose nearest)
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Click **Create Web Service**. Render will automatically install dependencies, initialize the database, and assign a live public HTTPS URL.

---

## 8. Academic Credits & License
- Developed as a B.Tech AI/ML Engineering Project.
- Mapping Data: © OpenStreetMap contributors.
- Telemetry: Open-Meteo & Central Water Commission / IMD Historical monsoon baselines.
