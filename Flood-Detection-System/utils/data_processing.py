"""
=============================================================================
FLOOD DETECTION SYSTEM - DATA PROCESSING UTILITIES
Location references, input validation, and feature preparation.
=============================================================================
"""

import pandas as pd
import numpy as np

# Canonical list of 7 verified Punjab locations with geographical references
LOCATIONS_DATA = {
    "Jalandhar": {
        "name": "Jalandhar",
        "district": "Jalandhar",
        "latitude": 31.3260,
        "longitude": 75.5762,
        "elevation_m": 228.0,
        "nearest_river": "Kali Bein / Sutlej tributary",
        "distance_to_river_km": 12.0,
        "normal_river_level_m": 2.2,
        "danger_river_level_m": 4.5,
        "description": "Commercial hub in Doaba plain between Beas and Sutlej rivers.",
        # Regional baseline seasonal norms (used when offline/fallback)
        "baseline": {
            "rainfall_mm": 5.2,
            "temp_c": 28.5,
            "humidity_pct": 58.0,
            "river_water_level_m": 2.1,
            "soil_moisture_pct": 36.0,
            "prev_rainfall_7d_mm": 18.0
        }
    },
    "Phagwara": {
        "name": "Phagwara",
        "district": "Kapurthala",
        "latitude": 31.2240,
        "longitude": 75.7708,
        "elevation_m": 234.0,
        "nearest_river": "East Bein drainage basin",
        "distance_to_river_km": 8.5,
        "normal_river_level_m": 2.0,
        "danger_river_level_m": 4.2,
        "description": "Industrial town in central Doaba; drained by seasonal East Bein rivulets.",
        "baseline": {
            "rainfall_mm": 4.8,
            "temp_c": 29.0,
            "humidity_pct": 56.0,
            "river_water_level_m": 1.9,
            "soil_moisture_pct": 34.0,
            "prev_rainfall_7d_mm": 15.0
        }
    },
    "Amritsar": {
        "name": "Amritsar",
        "district": "Amritsar",
        "latitude": 31.6340,
        "longitude": 74.8723,
        "elevation_m": 234.0,
        "nearest_river": "Ravi River / Sakki Nullah",
        "distance_to_river_km": 15.0,
        "normal_river_level_m": 2.4,
        "danger_river_level_m": 4.8,
        "description": "Historic Majha city bounded by Ravi River to the northwest and Sakki Nullah.",
        "baseline": {
            "rainfall_mm": 3.6,
            "temp_c": 27.8,
            "humidity_pct": 52.0,
            "river_water_level_m": 2.2,
            "soil_moisture_pct": 32.0,
            "prev_rainfall_7d_mm": 12.0
        }
    },
    "Ludhiana": {
        "name": "Ludhiana",
        "district": "Ludhiana",
        "latitude": 30.9010,
        "longitude": 75.8573,
        "elevation_m": 244.0,
        "nearest_river": "Sutlej River & Buddha Nullah",
        "distance_to_river_km": 6.0,
        "normal_river_level_m": 2.6,
        "danger_river_level_m": 5.0,
        "description": "Major metropolis along the southern banks of Sutlej with internal Buddha Nullah drainage.",
        "baseline": {
            "rainfall_mm": 6.5,
            "temp_c": 29.5,
            "humidity_pct": 60.0,
            "river_water_level_m": 2.5,
            "soil_moisture_pct": 38.0,
            "prev_rainfall_7d_mm": 22.0
        }
    },
    "Kapurthala": {
        "name": "Kapurthala",
        "district": "Kapurthala",
        "latitude": 31.3800,
        "longitude": 75.3810,
        "elevation_m": 225.0,
        "nearest_river": "Beas River & Harike Wetland",
        "distance_to_river_km": 7.0,
        "normal_river_level_m": 2.5,
        "danger_river_level_m": 4.4,
        "description": "Low-lying floodplain adjacent to Beas River upstream of Harike confluence (Mand area).",
        "baseline": {
            "rainfall_mm": 5.8,
            "temp_c": 28.0,
            "humidity_pct": 62.0,
            "river_water_level_m": 2.4,
            "soil_moisture_pct": 42.0,
            "prev_rainfall_7d_mm": 20.0
        }
    },
    "Hoshiarpur": {
        "name": "Hoshiarpur",
        "district": "Hoshiarpur",
        "latitude": 31.5143,
        "longitude": 75.9115,
        "elevation_m": 296.0,
        "nearest_river": "Shivalik Seasonal Torrents (Choes)",
        "distance_to_river_km": 4.0,
        "normal_river_level_m": 1.5,
        "danger_river_level_m": 3.8,
        "description": "Foothill district prone to rapid torrential runoff (choes) from Shivalik ranges.",
        "baseline": {
            "rainfall_mm": 8.0,
            "temp_c": 26.5,
            "humidity_pct": 64.0,
            "river_water_level_m": 1.6,
            "soil_moisture_pct": 40.0,
            "prev_rainfall_7d_mm": 28.0
        }
    },
    "Nawanshahr": {
        "name": "Nawanshahr",
        "district": "Shaheed Bhagat Singh Nagar",
        "latitude": 31.1240,
        "longitude": 76.1160,
        "elevation_m": 250.0,
        "nearest_river": "Sutlej River floodplain",
        "distance_to_river_km": 5.5,
        "normal_river_level_m": 2.3,
        "danger_river_level_m": 4.7,
        "description": "Flanked by the Sutlej River dykes downriver from the Bhakra/Ropar barrage complex.",
        "baseline": {
            "rainfall_mm": 6.0,
            "temp_c": 28.2,
            "humidity_pct": 59.0,
            "river_water_level_m": 2.2,
            "soil_moisture_pct": 37.0,
            "prev_rainfall_7d_mm": 21.0
        }
    }
}

FEATURE_COLUMNS = [
    "rainfall_mm",
    "temp_c",
    "humidity_pct",
    "river_water_level_m",
    "soil_moisture_pct",
    "prev_rainfall_7d_mm",
    "elevation_m",
    "distance_to_river_km"
]

RISK_METADATA = {
    0: {
        "label": "Low",
        "color": "#16a34a",       # Forest green
        "badge_class": "risk-low",
        "summary": "Normal Hydrological Conditions",
        "advisory": "Water levels are safe and well within natural drainage capacity. Standard routine monitoring is sufficient.",
        "action": "Routine observation."
    },
    1: {
        "label": "Moderate",
        "color": "#ca8a04",       # Amber yellow
        "badge_class": "risk-moderate",
        "summary": "Elevated Hydrological Risk / Waterlogging",
        "advisory": "Localized surface waterlogging expected in low-lying roads and agricultural depressions. Drains are reaching capacity.",
        "action": "Inspect municipal drains and culverts. Alert local pump stations."
    },
    2: {
        "label": "High",
        "color": "#ea580c",       # Vivid orange
        "badge_class": "risk-high",
        "summary": "High Flood Hazard Alert",
        "advisory": "River water level approaching warning stage. Significant overland runoff with inundation risk along riverbanks.",
        "action": "Issue public warnings in riverside communities. Prepare sandbags and flood relief equipment."
    },
    3: {
        "label": "Severe",
        "color": "#dc2626",       # Deep red
        "badge_class": "risk-severe",
        "summary": "Critical Inundation / Embankment Danger",
        "advisory": "Danger stage exceeded. High probability of river embankment breach and severe residential/crop inundation.",
        "action": "Initiate immediate evacuation protocols in vulnerable floodplain zones. Dispatch disaster management teams."
    }
}


def sanitize_input_features(raw_data, location_name="Jalandhar"):
    """
    Validate, sanitize, and impute missing environmental parameters.
    """
    loc_meta = LOCATIONS_DATA.get(location_name, LOCATIONS_DATA["Jalandhar"])

    def _clean_val(key, default_val, min_val, max_val):
        val = raw_data.get(key)
        if val is None or val == "":
            val = default_val
        try:
            val = float(val)
        except (ValueError, TypeError):
            val = float(default_val)
        return float(np.clip(val, min_val, max_val))

    rainfall = _clean_val("rainfall_mm", loc_meta["baseline"]["rainfall_mm"], 0.0, 500.0)
    temp = _clean_val("temp_c", loc_meta["baseline"]["temp_c"], 0.0, 55.0)
    humidity = _clean_val("humidity_pct", loc_meta["baseline"]["humidity_pct"], 10.0, 100.0)
    river_level = _clean_val("river_water_level_m", loc_meta["baseline"]["river_water_level_m"], 0.0, 15.0)
    soil_moisture = _clean_val("soil_moisture_pct", loc_meta["baseline"]["soil_moisture_pct"], 5.0, 100.0)
    prev_rain = _clean_val("prev_rainfall_7d_mm", loc_meta["baseline"]["prev_rainfall_7d_mm"], 0.0, 800.0)
    elevation = _clean_val("elevation_m", loc_meta["elevation_m"], 100.0, 800.0)
    distance = _clean_val("distance_to_river_km", loc_meta["distance_to_river_km"], 0.1, 100.0)

    clean_dict = {
        "rainfall_mm": round(rainfall, 2),
        "temp_c": round(temp, 1),
        "humidity_pct": round(humidity, 1),
        "river_water_level_m": round(river_level, 2),
        "soil_moisture_pct": round(soil_moisture, 1),
        "prev_rainfall_7d_mm": round(prev_rain, 2),
        "elevation_m": round(elevation, 1),
        "distance_to_river_km": round(distance, 1)
    }

    return clean_dict


def features_to_dataframe(features_dict):
    """Convert cleaned feature dictionary to single-row DataFrame."""
    return pd.DataFrame([features_dict], columns=FEATURE_COLUMNS)
