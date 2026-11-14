"""
=============================================================================
FLOOD DETECTION SYSTEM - INTEGRATED AUTOMATED TEST SUITE
Validates ML model loading, predictions, authentication, and SQLite storage.
=============================================================================
"""

import os
import sys
import unittest
import json

# Add parent directory to sys.path so test imports work
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from app import app
from utils.prediction import load_model, predict_flood_risk, load_model_metadata
from utils.weather_service import fetch_environmental_data
from utils.data_processing import LOCATIONS_DATA
from database.db import (
    init_db,
    save_prediction,
    get_recent_predictions,
    authenticate_user,
    register_user
)


class TestFloodDetectionSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Prepare Flask test client and ensure DB is initialized."""
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        cls.client = app.test_client()
        init_db()

    # -----------------------------------------------------------------------
    # 1. Machine Learning Model Unit Tests
    # -----------------------------------------------------------------------

    def test_01_model_loading(self):
        """Test that the trained Random Forest model artifact loads correctly."""
        model = load_model()
        self.assertIsNotNone(model, "Model pipeline should not be None")
        self.assertTrue(hasattr(model, "predict"), "Model must implement predict method")
        self.assertTrue(hasattr(model, "predict_proba"), "Model must implement predict_proba method")

    def test_02_model_metadata(self):
        """Test that model evaluation metadata is present and valid."""
        meta = load_model_metadata()
        self.assertIn("metrics", meta)
        self.assertIn("accuracy", meta["metrics"])
        self.assertGreaterEqual(meta["metrics"]["accuracy"], 0.90, "Model accuracy should exceed 90%")
        self.assertIn("confusion_matrix", meta)
        self.assertIn("feature_importances", meta)

    def test_03_prediction_dry_scenario(self):
        """Test prediction for clear/dry scenario yields Low Risk."""
        dry_input = {
            "rainfall_mm": 2.0,
            "temp_c": 32.0,
            "humidity_pct": 40.0,
            "river_water_level_m": 1.5,
            "soil_moisture_pct": 25.0,
            "prev_rainfall_7d_mm": 5.0,
            "elevation_m": 228.0,
            "distance_to_river_km": 12.0
        }
        res = predict_flood_risk(dry_input, location_name="Jalandhar")
        self.assertEqual(res["prediction"]["risk_level"], 0, "Dry conditions must classify as Low Risk (0)")
        self.assertEqual(res["prediction"]["risk_label"], "Low")
        self.assertGreater(res["prediction"]["confidence_pct"], 50.0)

    def test_04_prediction_severe_flood_scenario(self):
        """Test prediction for catastrophic rainfall & river surge yields Severe Risk."""
        severe_input = {
            "rainfall_mm": 190.0,
            "temp_c": 24.0,
            "humidity_pct": 98.0,
            "river_water_level_m": 6.2,
            "soil_moisture_pct": 98.0,
            "prev_rainfall_7d_mm": 260.0,
            "elevation_m": 225.0,
            "distance_to_river_km": 4.0
        }
        res = predict_flood_risk(severe_input, location_name="Kapurthala")
        self.assertEqual(res["prediction"]["risk_level"], 3, "Catastrophic event must classify as Severe Risk (3)")
        self.assertEqual(res["prediction"]["risk_label"], "Severe")
        self.assertGreater(res["prediction"]["confidence_pct"], 60.0)

    # -----------------------------------------------------------------------
    # 2. Location & Environmental Data Service Tests
    # -----------------------------------------------------------------------

    def test_05_verified_locations(self):
        """Test that all 7 required Punjab locations exist with authentic coordinates."""
        expected_cities = [
            "Jalandhar", "Phagwara", "Amritsar", "Ludhiana",
            "Kapurthala", "Hoshiarpur", "Nawanshahr"
        ]
        for city in expected_cities:
            self.assertIn(city, LOCATIONS_DATA, f"{city} must exist in catalog")
            loc = LOCATIONS_DATA[city]
            self.assertGreater(loc["latitude"], 30.0, f"{city} latitude within Punjab")
            self.assertLess(loc["latitude"], 33.0, f"{city} latitude within Punjab")
            self.assertGreater(loc["longitude"], 74.0, f"{city} longitude within Punjab")
            self.assertLess(loc["longitude"], 77.0, f"{city} longitude within Punjab")

    def test_06_weather_service_fallback(self):
        """Test weather service returns complete data even if offline."""
        data = fetch_environmental_data("Amritsar", timeout_seconds=1.0)
        self.assertIn("rainfall_mm", data)
        self.assertIn("temp_c", data)
        self.assertIn("humidity_pct", data)
        self.assertIn("river_water_level_m", data)
        self.assertIn("data_source", data)

    # -----------------------------------------------------------------------
    # 3. SQLite Database & Authentication Tests
    # -----------------------------------------------------------------------

    def test_07_database_insertion_and_query(self):
        """Test saving and querying predictions from SQLite."""
        sample_rec = {
            "location_name": "Ludhiana",
            "latitude": 30.9010,
            "longitude": 75.8573,
            "rainfall_mm": 45.0,
            "temp_c": 29.0,
            "humidity_pct": 75.0,
            "river_water_level_m": 3.2,
            "soil_moisture_pct": 60.0,
            "prev_rainfall_7d_mm": 55.0,
            "elevation_m": 244.0,
            "distance_to_river_km": 6.0,
            "risk_level": 1,
            "risk_label": "Moderate",
            "confidence_pct": 82.5,
            "data_source": "Unit Test Record"
        }
        rec_id = save_prediction(sample_rec)
        self.assertIsInstance(rec_id, int)
        self.assertGreater(rec_id, 0)

        recent = get_recent_predictions(limit=5)
        self.assertGreater(len(recent), 0)
        self.assertEqual(recent[0]["location_name"], "Ludhiana")

    def test_08_user_authentication(self):
        """Test authenticating default email user credentials."""
        user = authenticate_user("user@floodsystem.com", "password123")
        self.assertIsNotNone(user)
        self.assertEqual(user["email"], "user@floodsystem.com")

        invalid_user = authenticate_user("user@floodsystem.com", "wrongpass")
        self.assertIsNone(invalid_user)

    # -----------------------------------------------------------------------
    # 4. Flask HTTP View, Auth Routes & REST API Tests
    # -----------------------------------------------------------------------

    def test_09_view_routes(self):
        """Test that web pages respond with HTTP 200."""
        routes = ["/", "/model", "/history", "/login", "/register"]
        for r in routes:
            response = self.client.get(r)
            self.assertEqual(response.status_code, 200, f"Route {r} should return 200")
            self.assertIn(b"FLOOD DETECTION SYSTEM", response.data)

    def test_10_login_and_logout_flow(self):
        """Test logging in via POST /login using email and logging out."""
        # Successful login
        post_res = self.client.post("/login", data={
            "email": "user@floodsystem.com",
            "password": "password123"
        }, follow_redirects=True)
        self.assertEqual(post_res.status_code, 200)
        self.assertIn(b"Demo User", post_res.data)

        # Logout
        logout_res = self.client.get("/logout", follow_redirects=True)
        self.assertEqual(logout_res.status_code, 200)
        self.assertIn(b"Account Login", logout_res.data)

    def test_11_api_locations(self):
        """Test GET /api/locations endpoint."""
        response = self.client.get("/api/locations")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data["success"])
        self.assertEqual(data["count"], 7)

    def test_12_api_predict(self):
        """Test POST /api/predict endpoint."""
        payload = {
            "location_name": "Phagwara",
            "rainfall_mm": 85.0,
            "temp_c": 27.0,
            "humidity_pct": 88.0,
            "river_water_level_m": 3.9,
            "soil_moisture_pct": 80.0,
            "prev_rainfall_7d_mm": 110.0,
            "elevation_m": 234.0,
            "distance_to_river_km": 8.5,
            "data_source": "API Automated Test"
        }
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        res_json = json.loads(response.data)
        self.assertTrue(res_json["success"])
        self.assertIn("prediction", res_json["result"])
        self.assertIn(res_json["result"]["prediction"]["risk_label"], ["Low", "Moderate", "High", "Severe"])

    def test_13_api_map_markers(self):
        """Test GET /api/map-markers endpoint returns 7 populated markers."""
        response = self.client.get("/api/map-markers")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data["success"])
        self.assertEqual(len(data["markers"]), 7)


if __name__ == "__main__":
    unittest.main()
