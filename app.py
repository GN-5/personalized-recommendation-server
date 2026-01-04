from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os
import joblib
import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import credentials, auth, firestore
from functools import wraps

from ingest_processing import process_database 

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# Firebase setup
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-admin-config.json")
    firebase_admin.initialize_app(cred)

db_firestore = firestore.client()

# Load Random Forest model
MODEL_PATH = os.path.join(os.getcwd(), "models", "rf_model.pkl")
rf = joblib.load(MODEL_PATH)

def verify_firebase_token(token):
    print("Verifying token:", token)
    test_users = {
        "TEST": {"uid": "test_user", "email": "test@example.com", "name": "TestUser"},
    }
    if token in test_users:
        return test_users[token]
    try:
        decoded_token = auth.verify_id_token(token)
        return {
            "uid": decoded_token["uid"],
            "email": decoded_token.get("email"),
            "name": decoded_token.get("name", "")
        }
    except Exception as e:
        raise Exception("Invalid or unverified token.")

def require_auth(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        token = auth_header.split("Bearer ")[1]
        try:
            user = verify_firebase_token(token)
        except Exception as e:
            return jsonify({"error": str(e)}), 401
        kwargs["user"] = user
        return func(*args, **kwargs)
    return wrapper

def predict_steps(features):
    """
    Predict steps using the RandomForest model given a list of feature lists.
    Each feature list should be in the form:
      [total_sleep_minutes, day_of_week, prev_sleep, prev_steps]
    """
    X = np.array(features)
    preds = rf.predict(X)
    return preds

@app.route('/ingest', methods=['POST'])
@require_auth
def ingest(user):
    username = user.get("name")
    if not username:
        return jsonify({"error": "Username is required for processing"}), 400

    doc_ref = db_firestore.collection("user_history").document(username)
    if doc_ref.get().exists:
        return jsonify({"status": "Data already ingested"}), 200

    data_dir = os.path.join(os.getcwd(), "data")
    db_path = os.path.join(data_dir, f"healthkit{username}.db")
    if not os.path.exists(db_path):
        return jsonify({"error": "Database not found for user"}), 404

    df_daily = process_database(db_path)
    if df_daily is None:
        return jsonify({"error": "Processing failed"}), 500

    records = df_daily.to_dict(orient="records")
    for rec in records:
        date_str = rec["date"].strftime("%Y-%m-%d") if isinstance(rec["date"], pd.Timestamp) else str(rec["date"])
        db_firestore.collection("user_history").document(username).collection("history").document(date_str).set(rec)

    return jsonify({"status": "Data ingested to Firestore"}), 200

@app.route('/predict', methods=['POST'])
@require_auth
def predict(user):
    data = request.get_json()
    features = data.get("features")
    if not features:
        return jsonify({"error": "No features provided"}), 400
    try:
        preds = predict_steps(features)
        return jsonify({"predictions": preds.tolist()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict-public', methods=['POST'])
def predict_public():
    data = request.get_json()
    features = data.get("features")
    if not features:
        return jsonify({"error": "No features provided"}), 400
    try:
        preds = predict_steps(features)
        return jsonify({"predictions": preds.tolist()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/history', methods=['GET'])
@require_auth
def history(user):
    username = user.get("name")
    try:
        docs = db_firestore.collection("user_history").document(username).collection("history").stream()
        records = [doc.to_dict() for doc in docs]
        return jsonify({"history": sorted(records, key=lambda x: x["date"])}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


import datetime

@app.route('/dashboard', methods=['GET'])
@require_auth
def dashboard(user):
    username = user.get("name")
    try:
        # Fetch user history
        docs = db_firestore.collection("user_history").document(username).collection("history").stream()
        records = []
        for doc in docs:
            data = doc.to_dict()
            # Convert Firestore timestamp to string format
            if isinstance(data.get("date"), datetime.datetime):
                data["date"] = data["date"].strftime("%Y-%m-%d")
            records.append(data)

        if not records:
            return jsonify({"dashboard": {}}), 200

        # Sort by date string and get the latest
        latest = sorted(records, key=lambda x: x["date"], reverse=True)[0]
        date_str = latest["date"]

        # Compute day_of_week
        try:
            day_of_week = datetime.datetime.strptime(date_str, "%Y-%m-%d").weekday()
        except Exception:
            return jsonify({"error": "Invalid date format"}), 400

        # Prepare features for prediction
        features = [[
            latest.get("total_sleep_minutes", 0),
            day_of_week,
            latest.get("prev_sleep", 0),
            latest.get("prev_steps", 0)
        ]]

        # Predict
        predicted_steps = predict_steps(features)[0]
        latest["predicted_steps"] = round(predicted_steps, 2)

        # Update Firestore
        db_firestore.collection("user_history") \
            .document(username) \
            .collection("history") \
            .document(date_str).update({"predicted_steps": latest["predicted_steps"]})

        return jsonify({"dashboard": latest}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/append', methods=['POST'])
@require_auth
def append(user):
    username = user.get("name")
    data = request.get_json()
    required_fields = ["date", "prev_sleep", "prev_steps"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400
    try:
        date_str = data["date"]
        db_firestore.collection("user_history").document(username).collection("history").document(date_str).set(data)
        return jsonify({"status": "Record added"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/update', methods=['POST'])
@require_auth
def update(user):
    username = user.get("name")
    data = request.get_json()
    date = data.get("date")
    predicted_steps = data.get("predicted_steps")

    if not date or predicted_steps is None:
        return jsonify({"error": "Both 'date' and 'predicted_steps' are required"}), 400

    try:
        db_firestore.collection("user_history").document(username)\
            .collection("history").document(date)\
            .update({"predicted_steps": predicted_steps})
        return jsonify({"status": "Prediction updated"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=3001)
