"""
UPI Fraud Detection - Flask Backend
Endpoints:
  POST /analyze  - analyze a transaction, store in DB, return ML + rule-based prediction
  GET  /history  - retrieve transaction history from DB
  GET  /stats    - aggregate stats for dashboard
"""

from flask import Flask, request, jsonify, render_template
import sqlite3
import pickle
import os
import json
from datetime import datetime

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "transactions.db")
MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model.pkl")

# ── Load ML model ─────────────────────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)
ml_model = model_data["model"]
ml_features = model_data["features"]
print("ML model loaded successfully.")

# ── Database setup ────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            sender_upi TEXT NOT NULL,
            receiver_upi TEXT NOT NULL,
            amount REAL NOT NULL,
            hour INTEGER NOT NULL,
            freq_today INTEGER NOT NULL,
            device_match INTEGER NOT NULL,
            receiver_type INTEGER NOT NULL,
            rule_score INTEGER NOT NULL,
            ml_fraud_prob REAL NOT NULL,
            final_risk_level TEXT NOT NULL,
            status TEXT NOT NULL,
            flags TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── Rule-based scoring engine ─────────────────────────────────────────────────
def rule_based_score(data):
    score = 0
    flags = []

    hour = data["hour"]
    if 0 <= hour < 5:
        score += 20
        flags.append({"text": "Transaction between midnight and 5am", "level": "high"})
    elif hour >= 22 or hour < 7:
        score += 10
        flags.append({"text": "Transaction outside typical business hours", "level": "medium"})
    else:
        flags.append({"text": "Transaction during normal hours", "level": "low"})

    amount = data["amount"]
    if amount > 100000:
        score += 25
        flags.append({"text": f"Very large amount ₹{amount:,.0f}", "level": "high"})
    elif amount > 25000:
        score += 12
        flags.append({"text": f"Above-average amount ₹{amount:,.0f}", "level": "medium"})
    else:
        flags.append({"text": f"Normal amount ₹{amount:,.0f}", "level": "low"})

    freq = data["freq_today"]
    if freq >= 11:
        score += 25
        flags.append({"text": f"{freq} transactions today — very high frequency", "level": "high"})
    elif freq >= 6:
        score += 12
        flags.append({"text": f"{freq} transactions today — elevated frequency", "level": "medium"})
    else:
        flags.append({"text": f"{freq} transaction(s) today — normal frequency", "level": "low"})

    device = data["device_match"]
    device_map = {0: ("low", "Known device and location", 0),
                  1: ("medium", "New device, known location", 14),
                  2: ("medium", "Known device, new location", 10),
                  3: ("high", "New device and new location", 25)}
    lv, txt, pts = device_map.get(device, ("low", "Known device", 0))
    score += pts
    flags.append({"text": txt, "level": lv})

    recv = data["receiver_type"]
    recv_map = {0: ("low", "Known/saved contact", 0),
                1: ("low", "Registered merchant", 0),
                2: ("medium", "First-time receiver", 8),
                3: ("high", "Suspicious/unverified receiver", 20)}
    lv, txt, pts = recv_map.get(recv, ("low", "Known contact", 0))
    score += pts
    flags.append({"text": txt, "level": lv})

    if not data.get("sender_id_valid", True):
        score += 5
        flags.append({"text": "Sender UPI ID format looks unusual", "level": "medium"})
    if not data.get("receiver_id_valid", True):
        score += 5
        flags.append({"text": "Receiver UPI ID format looks unusual", "level": "medium"})

    return min(score, 100), flags

# ── ML prediction ─────────────────────────────────────────────────────────────
def ml_predict(data):
    import pandas as pd
    features = pd.DataFrame([{
        "hour": data["hour"],
        "amount": data["amount"],
        "freq_today": data["freq_today"],
        "device_match": data["device_match"],
        "receiver_type": data["receiver_type"],
        "sender_id_valid": int(data.get("sender_id_valid", True)),
        "receiver_id_valid": int(data.get("receiver_id_valid", True)),
    }])
    prob = ml_model.predict_proba(features)[0][1]
    return round(prob, 4)

# ── Risk level helper ─────────────────────────────────────────────────────────
def compute_risk_level(rule_score, ml_prob):
    # Combine rule score and ML probability (weighted)
    combined = (rule_score * 0.5) + (ml_prob * 100 * 0.5)
    if combined >= 60 or ml_prob >= 0.75:
        return "high"
    elif combined >= 35 or ml_prob >= 0.4:
        return "medium"
    return "low"

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    body = request.get_json(force=True)

    sender = body.get("sender_upi", "")
    receiver = body.get("receiver_upi", "")
    amount = float(body.get("amount", 0))
    hour = int(body.get("hour", 12))
    freq = int(body.get("freq_today", 1))
    device = int(body.get("device_match", 0))
    recv_type = int(body.get("receiver_type", 0))
    sender_valid = "@" in sender
    receiver_valid = "@" in receiver

    data = {
        "hour": hour,
        "amount": amount,
        "freq_today": freq,
        "device_match": device,
        "receiver_type": recv_type,
        "sender_id_valid": sender_valid,
        "receiver_id_valid": receiver_valid,
    }

    rule_score, flags = rule_based_score(data)
    ml_prob = ml_predict(data)
    risk_level = compute_risk_level(rule_score, ml_prob)
    auto_blocked = risk_level == "high"
    status = "Blocked" if auto_blocked else "Pending"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO transactions
        (timestamp, sender_upi, receiver_upi, amount, hour, freq_today,
         device_match, receiver_type, rule_score, ml_fraud_prob,
         final_risk_level, status, flags)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (timestamp, sender, receiver, amount, hour, freq,
          device, recv_type, rule_score, ml_prob,
          risk_level, status, json.dumps(flags)))
    tx_id = c.lastrowid
    conn.commit()
    conn.close()

    return jsonify({
        "tx_id": tx_id,
        "rule_score": rule_score,
        "ml_fraud_prob": round(ml_prob * 100, 1),
        "risk_level": risk_level,
        "auto_blocked": auto_blocked,
        "flags": flags,
        "timestamp": timestamp
    })

@app.route("/update_status", methods=["POST"])
def update_status():
    body = request.get_json(force=True)
    tx_id = body.get("tx_id")
    status = body.get("status")
    if tx_id and status in ["Approved", "Blocked", "Overridden"]:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE transactions SET status=? WHERE id=?", (status, tx_id))
        conn.commit()
        conn.close()
    return jsonify({"ok": True})

@app.route("/history", methods=["GET"])
def history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, timestamp, sender_upi, receiver_upi, amount,
               rule_score, ml_fraud_prob, final_risk_level, status
        FROM transactions ORDER BY id DESC LIMIT 50
    """).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/stats", methods=["GET"])
def stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    total = c.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    blocked = c.execute("SELECT COUNT(*) FROM transactions WHERE final_risk_level='high'").fetchone()[0]
    avg_score = c.execute("SELECT AVG(rule_score) FROM transactions").fetchone()[0] or 0
    approved = c.execute("SELECT COUNT(*) FROM transactions WHERE status='Approved'").fetchone()[0]
    conn.close()
    return jsonify({
        "total": total,
        "blocked": blocked,
        "approved": approved,
        "avg_risk_score": round(avg_score, 1)
    })

if __name__ == "__main__":
    print("Starting UPI Fraud Detection server on http://localhost:5000")
    app.run(debug=True, port=5000)
