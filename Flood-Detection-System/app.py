"""
=============================================================================
FLOOD DETECTION SYSTEM - FLASK APPLICATION
Main Web Server, Authentication & API Controllers
=============================================================================
"""

import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from utils.data_processing import LOCATIONS_DATA, RISK_METADATA
from utils.weather_service import fetch_environmental_data
from utils.prediction import predict_flood_risk, load_model_metadata
from database.db import (
    init_db,
    save_prediction,
    get_recent_predictions,
    get_location_latest_predictions,
    authenticate_user,
    register_user,
    get_user_by_id
)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "punjab-flood-detection-secret-key-2024")

# Initialize SQLite database schema & default user accounts
init_db()


@app.route("/")
def index():
    """Render the primary interactive map and flood risk dashboard."""
    locations = list(LOCATIONS_DATA.values())
    default_city = "Jalandhar"
    
    # Pre-fetch initial environmental data for the default city
    initial_env = fetch_environmental_data(default_city)
    
    # Pre-calculate baseline risk prediction for default city
    initial_pred = predict_flood_risk(
        initial_env,
        location_name=default_city,
        data_source=initial_env["data_source"]
    )
    
    # Save the initial assessment
    try:
        db_record = {
            "location_name": default_city,
            "latitude": initial_pred["latitude"],
            "longitude": initial_pred["longitude"],
            "rainfall_mm": initial_pred["inputs"]["rainfall_mm"],
            "temp_c": initial_pred["inputs"]["temp_c"],
            "humidity_pct": initial_pred["inputs"]["humidity_pct"],
            "river_water_level_m": initial_pred["inputs"]["river_water_level_m"],
            "soil_moisture_pct": initial_pred["inputs"]["soil_moisture_pct"],
            "prev_rainfall_7d_mm": initial_pred["inputs"]["prev_rainfall_7d_mm"],
            "elevation_m": initial_pred["inputs"]["elevation_m"],
            "distance_to_river_km": initial_pred["inputs"]["distance_to_river_km"],
            "risk_level": initial_pred["prediction"]["risk_level"],
            "risk_label": initial_pred["prediction"]["risk_label"],
            "confidence_pct": initial_pred["prediction"]["confidence_pct"],
            "data_source": initial_env["data_source"]
        }
        save_prediction(db_record)
    except Exception as e:
        print(f"[DB Warning] Could not log initial prediction: {e}")

    return render_template(
        "index.html",
        locations=locations,
        default_city=default_city,
        initial_env=initial_env,
        initial_pred=initial_pred,
        risk_meta=RISK_METADATA
    )


@app.route("/model")
def model_info():
    """Render model architecture, evaluation metrics, and feature importance."""
    metadata = load_model_metadata()
    return render_template("model_info.html", metadata=metadata)


@app.route("/history")
def history():
    """Render the prediction history log stored in SQLite."""
    records = get_recent_predictions(limit=100)
    return render_template("history.html", records=records)


# ---------------------------------------------------------------------------
# Authentication Routes
# ---------------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle user email login."""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = authenticate_user(email, password)
        if user:
            session["user_id"] = user["id"]
            session["user_email"] = user["email"]
            session["user_name"] = user["name"]
            flash(f"Welcome back, {user['name']}!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid email or password. Please try again.", "danger")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Handle new user registration via email."""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        success, result = register_user(email, password, name)
        if success:
            flash("Account registered successfully! Please log in.", "success")
            return redirect(url_for("login"))
        else:
            flash(result, "danger")

    return render_template("register.html")


@app.route("/logout")
def logout():
    """Clear user session and redirect to login."""
    session.clear()
    flash("You have been securely logged out.", "info")
    return redirect(url_for("login"))


@app.route("/about")
def about():
    """Redirect about route cleanly to home."""
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/locations", methods=["GET"])
def api_locations():
    """Return catalog of Punjab monitoring locations."""
    return jsonify({
        "success": True,
        "count": len(LOCATIONS_DATA),
        "locations": LOCATIONS_DATA
    })


@app.route("/api/weather/<location_name>", methods=["GET"])
def api_weather(location_name):
    """Fetch live atmospheric data (with offline baseline fallback)."""
    if location_name not in LOCATIONS_DATA:
        return jsonify({"success": False, "error": f"Unknown location '{location_name}'"}), 404

    data = fetch_environmental_data(location_name)
    return jsonify({
        "success": True,
        "location": location_name,
        "data": data
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Execute ML prediction for given location and parameters, and record to DB.
    """
    req_json = request.get_json(silent=True) or {}
    location_name = req_json.get("location_name", "Jalandhar")
    data_source = req_json.get("data_source", "Manual Parameter Simulation")

    try:
        # Run inference through ML pipeline
        result = predict_flood_risk(req_json, location_name=location_name, data_source=data_source)

        # Persist record in SQLite
        db_record = {
            "location_name": location_name,
            "latitude": result["latitude"],
            "longitude": result["longitude"],
            "rainfall_mm": result["inputs"]["rainfall_mm"],
            "temp_c": result["inputs"]["temp_c"],
            "humidity_pct": result["inputs"]["humidity_pct"],
            "river_water_level_m": result["inputs"]["river_water_level_m"],
            "soil_moisture_pct": result["inputs"]["soil_moisture_pct"],
            "prev_rainfall_7d_mm": result["inputs"]["prev_rainfall_7d_mm"],
            "elevation_m": result["inputs"]["elevation_m"],
            "distance_to_river_km": result["inputs"]["distance_to_river_km"],
            "risk_level": result["prediction"]["risk_level"],
            "risk_label": result["prediction"]["risk_label"],
            "confidence_pct": result["prediction"]["confidence_pct"],
            "data_source": data_source
        }
        record_id = save_prediction(db_record)
        result["db_id"] = record_id

        return jsonify({"success": True, "result": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def api_history():
    """Return recent predictions as JSON."""
    limit = request.args.get("limit", default=30, type=int)
    records = get_recent_predictions(limit=limit)
    return jsonify({"success": True, "count": len(records), "records": records})


@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    """Return model evaluation metrics and feature importances."""
    metadata = load_model_metadata()
    return jsonify({"success": True, "metadata": metadata})


@app.route("/api/map-markers", methods=["GET"])
def api_map_markers():
    """
    Return current risk status and coordinates for all 7 Punjab stations
    to populate the Leaflet map.
    """
    latest_db_records = get_location_latest_predictions()
    markers = []

    for name, loc in LOCATIONS_DATA.items():
        if name in latest_db_records:
            rec = latest_db_records[name]
            risk_level = rec["risk_level"]
            risk_label = rec["risk_label"]
            confidence = rec["confidence_pct"]
            rainfall = rec["rainfall_mm"]
            humidity = rec["humidity_pct"]
            river_level = rec["river_water_level_m"]
            source = rec["data_source"]
        else:
            # Fall back to quick baseline prediction
            base_pred = predict_flood_risk(loc["baseline"], location_name=name, data_source="Regional Baseline")
            risk_level = base_pred["prediction"]["risk_level"]
            risk_label = base_pred["prediction"]["risk_label"]
            confidence = base_pred["prediction"]["confidence_pct"]
            rainfall = loc["baseline"]["rainfall_mm"]
            humidity = loc["baseline"]["humidity_pct"]
            river_level = loc["baseline"]["river_water_level_m"]
            source = "Regional Baseline"

        meta = RISK_METADATA.get(risk_level, RISK_METADATA[0])

        markers.append({
            "name": name,
            "district": loc["district"],
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "elevation_m": loc["elevation_m"],
            "nearest_river": loc["nearest_river"],
            "rainfall_mm": rainfall,
            "humidity_pct": humidity,
            "river_water_level_m": river_level,
            "danger_river_level_m": loc["danger_river_level_m"],
            "risk_level": risk_level,
            "risk_label": risk_label,
            "color": meta["color"],
            "confidence_pct": confidence,
            "data_source": source
        })

    return jsonify({"success": True, "markers": markers})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n===========================================================")
    print(f"Punjab Flood Detection System running at http://127.0.0.1:{port}")
    print(f"===========================================================\n")
    app.run(host="0.0.0.0", port=port, debug=True)
