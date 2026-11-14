"""
=============================================================================
FLOOD DETECTION SYSTEM - SQLITE DATABASE MODULE
Handles initialization, persistence, retrieval of predictions, and email authentication.
=============================================================================
"""

import os
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# Resolve database file path relative to this file
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "flood.db")


def get_db_connection():
    """Create and return a database connection with dict-like row access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables and seed default email account if not exists."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()

    # Predictions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            rainfall_mm REAL NOT NULL,
            temp_c REAL NOT NULL,
            humidity_pct REAL NOT NULL,
            river_water_level_m REAL NOT NULL,
            soil_moisture_pct REAL NOT NULL,
            prev_rainfall_7d_mm REAL NOT NULL,
            elevation_m REAL NOT NULL,
            distance_to_river_km REAL NOT NULL,
            risk_level INTEGER NOT NULL,
            risk_label TEXT NOT NULL,
            confidence_pct REAL NOT NULL,
            data_source TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Check if users table needs migration to email schema
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    if cursor.fetchone():
        cursor.execute("PRAGMA table_info(users)")
        cols = [col[1] for col in cursor.fetchall()]
        if "email" not in cols:
            # Drop old table to migrate to clean email schema
            cursor.execute("DROP TABLE users")

    # Users Table for Email Authentication
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Index for fast retrieval by location and recency
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_location 
        ON predictions(location_name, created_at DESC)
    """)

    # Seed default user account if table is empty
    cursor.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]

    if user_count == 0:
        default_pwd = generate_password_hash("password123")
        cursor.execute("""
            INSERT INTO users (email, name, password_hash)
            VALUES (?, ?, ?)
        """, ("user@floodsystem.com", "Demo User", default_pwd))

    conn.commit()
    conn.close()


def save_prediction(record):
    """
    Insert a prediction record into the SQLite database.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            location_name, latitude, longitude,
            rainfall_mm, temp_c, humidity_pct,
            river_water_level_m, soil_moisture_pct, prev_rainfall_7d_mm,
            elevation_m, distance_to_river_km,
            risk_level, risk_label, confidence_pct,
            data_source, created_at
        ) VALUES (
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?, ?,
            ?, ?
        )
    """, (
        record.get("location_name"),
        float(record.get("latitude", 0.0)),
        float(record.get("longitude", 0.0)),
        float(record.get("rainfall_mm", 0.0)),
        float(record.get("temp_c", 0.0)),
        float(record.get("humidity_pct", 0.0)),
        float(record.get("river_water_level_m", 0.0)),
        float(record.get("soil_moisture_pct", 0.0)),
        float(record.get("prev_rainfall_7d_mm", 0.0)),
        float(record.get("elevation_m", 0.0)),
        float(record.get("distance_to_river_km", 0.0)),
        int(record.get("risk_level", 0)),
        record.get("risk_label", "Low"),
        float(record.get("confidence_pct", 0.0)),
        record.get("data_source", "Manual Input"),
        record.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    ))

    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def get_recent_predictions(limit=50):
    """Retrieve recent predictions ordered by newest first."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    results = [dict(row) for row in rows]
    conn.close()
    return results


def get_location_latest_predictions():
    """Retrieve the latest prediction for each location."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT p.*
        FROM predictions p
        INNER JOIN (
            SELECT location_name, MAX(id) as max_id
            FROM predictions
            GROUP BY location_name
        ) latest ON p.id = latest.max_id
        ORDER BY p.location_name ASC
    """)

    rows = cursor.fetchall()
    results = {row["location_name"]: dict(row) for row in rows}
    conn.close()
    return results


# ---------------------------------------------------------------------------
# Email Authentication Helpers
# ---------------------------------------------------------------------------

def register_user(email, password, name=""):
    """
    Register a new user using email and password.
    Returns (success_bool, message_or_user_dict).
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    email = email.strip().lower()
    if not email or "@" not in email:
        conn.close()
        return False, "Please enter a valid email address."

    if not password or len(password) < 4:
        conn.close()
        return False, "Password must be at least 4 characters long."

    # Check if email is already registered
    cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
    if cursor.fetchone():
        conn.close()
        return False, f"The email '{email}' is already registered. Please log in."

    display_name = name.strip() if name and name.strip() else email.split("@")[0].title()
    pwd_hash = generate_password_hash(password)

    try:
        cursor.execute("""
            INSERT INTO users (email, name, password_hash)
            VALUES (?, ?, ?)
        """, (email, display_name, pwd_hash))
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return True, {"id": user_id, "email": email, "name": display_name}
    except Exception as e:
        conn.close()
        return False, str(e)


def authenticate_user(email, password):
    """
    Authenticate user with email and password.
    Returns user dict on success, None on failure.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    email = email.strip().lower()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user_row = cursor.fetchone()
    conn.close()

    if not user_row:
        return None

    if check_password_hash(user_row["password_hash"], password):
        return {
            "id": user_row["id"],
            "email": user_row["email"],
            "name": user_row["name"]
        }
    return None


def get_user_by_id(user_id):
    """Retrieve user record by ID."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, name, created_at FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None
