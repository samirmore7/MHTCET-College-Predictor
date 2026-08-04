import os
import gzip
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ==============================================================================
# 1. LOAD MODEL & ENCODERS (DIRECT GZIP IN-MEMORY DECOMPRESSION)
# ==============================================================================
MODEL_PATH = "institute_model_compressed.pkl.gz"
GENDER_PATH = "gender_encoder.pkl"
CATEGORY_PATH = "category_encoder.pkl"
SEAT_PATH = "seat_encoder.pkl"
COURSE_PATH = "course_encoder.pkl"
INSTITUTE_PATH = "institute_encoder.pkl"

def load_pickle(path):
    """Safely loads standard or gzipped pickle files."""
    if os.path.exists(path):
        if path.endswith(".gz"):
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        else:
            with open(path, "rb") as f:
                return pickle.load(f)
    return None

# Load model and encoders
model = load_pickle(MODEL_PATH)
gender_encoder = load_pickle(GENDER_PATH)
category_encoder = load_pickle(CATEGORY_PATH)
seat_encoder = load_pickle(SEAT_PATH)
course_encoder = load_pickle(COURSE_PATH)
institute_encoder = load_pickle(INSTITUTE_PATH)

# Helper to safely retrieve classes from LabelEncoders
def get_classes(encoder, fallback):
    if encoder and hasattr(encoder, 'classes_'):
        return list(encoder.classes_)
    return fallback

GENDER_OPTIONS = get_classes(gender_encoder, ["Female", "Male"])
CATEGORY_OPTIONS = get_classes(category_encoder, ["OPEN", "OBC", "SC", "ST", "SEBC", "NT 1 (NT-B)"])
SEAT_OPTIONS = get_classes(seat_encoder, ["GOPENS", "GSCS", "GSTS", "GOBCS", "LOPENS", "LSCS"])
INSTITUTE_OPTIONS = get_classes(institute_encoder, ["COEP Technological University", "VJTI Mumbai", "PICT Pune"])
COURSE_OPTIONS = get_classes(course_encoder, ["Computer Engineering", "Information Technology", "Artificial Intelligence"])

# ==============================================================================
# 2. FLASK API ROUTES
# ==============================================================================

@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        HTML_TEMPLATE,
        genders=GENDER_OPTIONS,
        categories=CATEGORY_OPTIONS,
        seats=SEAT_OPTIONS,
        institutes=INSTITUTE_OPTIONS
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        gender_str = data.get("gender")
        category_str = data.get("category")
        percentile = float(data.get("percentile", 0.0))
        seat_str = data.get("seat")
        institute_str = data.get("institute")

        # Encode string inputs back into categorical indices for the model
        gender_encoded = int(gender_encoder.transform([gender_str])[0]) if gender_encoder else 0
        category_encoded = int(category_encoder.transform([category_str])[0]) if category_encoder else 0
        seat_encoded = int(seat_encoder.transform([seat_str])[0]) if seat_encoder else 0
        institute_encoded = int(institute_encoder.transform([institute_str])[0]) if institute_encoder else 0

        # Feature vector matching trained feature order:
        # ['Gender', 'Category', 'MHTCET Percentile', 'Seat Alloted', 'Institute Name']
        features = np.array([[gender_encoded, category_encoded, percentile, seat_encoded, institute_encoded]])

        if model is not None:
            # Predict top class index
            pred_idx = model.predict(features)[0]
            
            # Convert class index back to readable Branch name
            if course_encoder and hasattr(course_encoder, 'inverse_transform'):
                predicted_course = course_encoder.inverse_transform([pred_idx])[0]
            else:
                predicted_course = COURSE_OPTIONS[int(pred_idx) % len(COURSE_OPTIONS)]
            
            # Predict class probabilities for confidence distribution
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                top_indices = np.argsort(probs)[::-1][:5]
                top_branches = []
                for idx in top_indices:
                    if probs[idx] > 0.001:
                        b_name = course_encoder.inverse_transform([idx])[0] if course_encoder else f"Branch {idx}"
                        top_branches.append({
                            "branch": b_name, 
                            "prob": round(float(probs[idx]) * 100, 2)
                        })
            else:
                top_branches = [{"branch": predicted_course, "prob": 98.5}]
        else:
            predicted_course = "Model file not found"
            top_branches = []

        return jsonify({
            "status": "success",
            "prediction": predicted_course,
            "top_branches": top_branches,
            "inputs": {
                "Gender": gender_str,
                "Category": category_str,
                "Percentile": percentile,
                "Seat Alloted": seat_str,
                "Institute": institute_str
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ==============================================================================
# 3. FRONTEND UI TEMPLATE (HTML + GLASSMORPHISM CSS + JS)
# ==============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="cyberpunk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Admission Allocation & Intelligence Engine</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        :root {
            --transition-speed: 0.4s;
            --radius-lg: 24px;
            --radius-md: 14px;
            --font-main: 'Plus Jakarta Sans', sans-serif;
            --font-heading: 'Space Grotesk', sans-serif;
        }

        [data-theme="cyberpunk"] {
            --bg-base: #060913;
            --bg-glass: rgba(15, 23, 42, 0.75);
            --bg-glass-hover: rgba(30, 41, 59, 0.85);
            --border-glass: rgba(255, 255, 255, 0.08);
            --border-accent: rgba(99, 102, 241, 0.4);
            --text-main: #F8FAFC;
            --text-muted: #94A3B8;
            --primary: #6366F1;
            --primary-glow: rgba(99, 102, 241, 0.45);
            --secondary: #EC4899;
            --accent: #06B6D4;
            --card-glow: 0 20px 50px rgba(0, 0, 0, 0.6);
        }

        [data-theme="gold"] {
            --bg-base: #0B0A07;
            --bg-glass: rgba(26, 22, 16, 0.75);
            --bg-glass-hover: rgba(45, 38, 28, 0.85);
            --border-glass: rgba(234, 179, 8, 0.15);
            --border-accent: rgba(234, 179, 8, 0.5);
            --text-main: #FEF08A;
            --text-muted: #CA8A04;
            --primary: #EAB308;
            --primary-glow: rgba(234, 179, 8, 0.45);
            --secondary: #F97316;
            --accent: #FACC15;
            --card-glow: 0 20px 50px rgba(0, 0, 0, 0.8);
        }

        [data-theme="light"] {
            --bg-base: #F1F5F9;
            --bg-glass: rgba(255, 255, 255, 0.8);
            --bg-glass-hover: rgba(255, 255, 255, 0.95);
            --border-glass: rgba(0, 0, 0, 0.08);
            --border-accent: rgba(99, 102, 241, 0.25);
            --text-main: #0F172A;
            --text-muted: #64748B;
            --primary: #4F46E5;
            --primary-glow: rgba(79, 70, 229, 0.25);
            --secondary: #DB2777;
            --accent: #0891B2;
            --card-glow: 0 20px 40px rgba(148, 163, 184, 0.25);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            transition: background-color var(--transition-speed), border-color var(--transition-speed), color var(--transition-speed);
        }

        body {
            font-family: var(--font-main);
            background-color: var(--bg-base);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 15% 15%, var(--primary-glow) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(236, 72, 153, 0.15) 0%, transparent 40%);
            background-attachment: fixed;
        }

        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 5%;
            backdrop-filter: blur(16px);
            background: var(--bg-glass);
            border-bottom: 1px solid var(--border-glass);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo {
            font-family: var(--font-heading);
            font-size: 1.5rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 12px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .theme-switcher {
            display: flex;
            background: rgba(0, 0, 0, 0.2);
            padding: 4px;
            border-radius: 30px;
            border: 1px solid var(--border-glass);
        }

        .theme-btn {
            border: none;
            background: transparent;
            color: var(--text-muted);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.3s ease;
        }

        .theme-btn.active {
            background: var(--primary);
            color: #FFF;
            box-shadow: 0 4px 15px var(--primary-glow);
        }

        .container {
            max-width: 1350px;
            margin: 40px auto;
            padding: 0 20px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 35px;
            flex: 1;
        }

        @media (max-width: 992px) {
            .container {
                grid-template-columns: 1fr;
            }
        }

        .glass-card {
            background: var(--bg-glass);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border-glass);
            border-radius: var(--radius-lg);
            padding: 35px;
            box-shadow: var(--card-glow);
            position: relative;
            overflow: hidden;
        }

        .glass-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--primary), transparent);
        }

        .card-header {
            margin-bottom: 28px;
        }

        .card-header h2 {
            font-family: var(--font-heading);
            font-size: 1.6rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .card-header p {
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 6px;
        }

        .form-group {
            margin-bottom: 22px;
        }

        .form-group label {
            display: block;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-muted);
            margin-bottom: 8px;
        }

        .input-wrapper {
            position: relative;
            display: flex;
            align-items: center;
        }

        .input-wrapper i {
            position: absolute;
            left: 16px;
            color: var(--primary);
            font-size: 1.1rem;
        }

        .form-control {
            width: 100%;
            padding: 14px 16px 14px 48px;
            background: rgba(0, 0, 0, 0.15);
            border: 1px solid var(--border-glass);
            border-radius: var(--radius-md);
            color: var(--text-main);
            font-size: 0.95rem;
            font-family: var(--font-main);
            outline: none;
            appearance: none;
        }

        .form-control:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px var(--primary-glow);
            background: var(--bg-glass-hover);
        }

        select.form-control {
            background-image: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%236366F1' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e");
            background-repeat: no-repeat;
            background-position: right 16px center;
            background-size: 16px;
        }

        select.form-control option {
            background-color: #0F172A;
            color: #FFF;
        }

        .btn-premium {
            width: 100%;
            padding: 16px;
            border: none;
            border-radius: var(--radius-md);
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: #FFF;
            font-size: 1rem;
            font-weight: 700;
            font-family: var(--font-heading);
            cursor: pointer;
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 25px var(--primary-glow);
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin-top: 10px;
        }

        .btn-premium::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.3), transparent);
            transform: rotate(45deg) translateY(-100%);
            transition: transform 0.6s ease;
        }

        .btn-premium:hover {
            transform: translateY(-3px) scale(1.01);
            box-shadow: 0 15px 35px var(--primary-glow);
        }

        .btn-premium:hover::before {
            transform: rotate(45deg) translateY(100%);
        }

        .btn-premium:active {
            transform: translateY(1px);
        }

        .results-panel {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .prediction-hero {
            background: var(--bg-glass-hover);
            border: 1px solid var(--border-accent);
            border-radius: var(--radius-md);
            padding: 25px;
            text-align: center;
            margin-bottom: 25px;
            position: relative;
            box-shadow: inset 0 0 20px var(--primary-glow);
        }

        .prediction-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--accent);
            font-weight: 700;
            margin-bottom: 10px;
        }

        .prediction-value {
            font-family: var(--font-heading);
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--text-main);
            line-height: 1.3;
        }

        .analytics-section h3 {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 18px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .prob-item {
            margin-bottom: 16px;
        }

        .prob-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            margin-bottom: 6px;
        }

        .prob-name {
            font-weight: 600;
            color: var(--text-main);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 80%;
        }

        .prob-val {
            font-weight: 700;
            color: var(--accent);
        }

        .progress-bg {
            height: 10px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid var(--border-glass);
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--accent));
            border-radius: 10px;
            width: 0%;
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 25px;
        }

        .metric-card {
            background: rgba(0, 0, 0, 0.15);
            border: 1px solid var(--border-glass);
            padding: 16px;
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .metric-icon {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            background: var(--primary-glow);
            color: var(--primary);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
        }

        .metric-info label {
            font-size: 0.75rem;
            color: var(--text-muted);
            display: block;
        }

        .metric-info span {
            font-size: 0.9rem;
            font-weight: 700;
            color: var(--text-main);
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }

        .pulse-animation {
            animation: pulse 2.5s infinite ease-in-out;
        }

        .history-card {
            max-width: 1350px;
            margin: 0 auto 40px auto;
            width: calc(100% - 40px);
        }

        .history-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 0.85rem;
        }

        .history-table th, .history-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-glass);
        }

        .history-table th {
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        footer {
            text-align: center;
            padding: 25px;
            font-size: 0.85rem;
            color: var(--text-muted);
            border-top: 1px solid var(--border-glass);
            margin-top: auto;
        }
    </style>
</head>
<body>

    <!-- NAVBAR -->
    <nav class="navbar">
        <div class="logo">
            <i class="fa-solid fa-graduation-cap"></i>
            <span>EduPredict<span style="color:var(--secondary)">.AI</span></span>
        </div>
        <div class="theme-switcher">
            <button class="theme-btn active" onclick="setTheme('cyberpunk', this)">
                <i class="fa-solid fa-bolt"></i> Cyber
            </button>
            <button class="theme-btn" onclick="setTheme('gold', this)">
                <i class="fa-solid fa-crown"></i> Luxe
            </button>
            <button class="theme-btn" onclick="setTheme('light', this)">
                <i class="fa-solid fa-sun"></i> Light
            </button>
        </div>
    </nav>

    <!-- MAIN CONTAINER -->
    <div class="container">
        
        <!-- INPUT FORM CARD -->
        <div class="glass-card">
            <div class="card-header">
                <h2><i class="fa-solid fa-sliders" style="color:var(--primary)"></i> Candidate Profile</h2>
                <p>Configure candidate attributes to infer optimal branch allocation</p>
            </div>

            <form id="predictionForm" onsubmit="handlePredict(event)">
                
                <!-- Gender -->
                <div class="form-group">
                    <label>Gender Category</label>
                    <div class="input-wrapper">
                        <i class="fa-solid fa-venus-mars"></i>
                        <select class="form-control" id="gender" required>
                            {% for g in genders %}
                            <option value="{{ g }}">{{ g }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>

                <!-- Category -->
                <div class="form-group">
                    <label>Reservation Category</label>
                    <div class="input-wrapper">
                        <i class="fa-solid fa-layer-group"></i>
                        <select class="form-control" id="category" required>
                            {% for c in categories %}
                            <option value="{{ c }}">{{ c }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>

                <!-- MHTCET Percentile -->
                <div class="form-group">
                    <label>MHTCET Percentile Score</label>
                    <div class="input-wrapper">
                        <i class="fa-solid fa-chart-line"></i>
                        <input type="number" step="0.0001" min="0" max="100" class="form-control" id="percentile" placeholder="e.g. 98.4521" required>
                    </div>
                </div>

                <!-- Seat Alloted -->
                <div class="form-group">
                    <label>Seat Allocation Quota</label>
                    <div class="input-wrapper">
                        <i class="fa-solid fa-chair"></i>
                        <select class="form-control" id="seat" required>
                            {% for s in seats %}
                            <option value="{{ s }}">{{ s }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>

                <!-- Institute Name -->
                <div class="form-group">
                    <label>Target Institute Name</label>
                    <div class="input-wrapper">
                        <i class="fa-solid fa-building-columns"></i>
                        <select class="form-control" id="institute" required>
                            {% for inst in institutes %}
                            <option value="{{ inst }}">{{ inst }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>

                <!-- SUBMIT BUTTON -->
                <button type="submit" class="btn-premium" id="submitBtn">
                    <span>Execute Inference</span>
                    <i class="fa-solid fa-wand-magic-sparkles"></i>
                </button>
            </form>
        </div>

        <!-- RESULTS / ANALYTICS DASHBOARD -->
        <div class="glass-card results-panel">
            <div>
                <div class="card-header">
                    <h2><i class="fa-solid fa-chart-pie" style="color:var(--accent)"></i> Allocation Dashboard</h2>
                    <p>Machine learning classification results and confidence metrics</p>
                </div>

                <!-- PREDICTED HERO CARD -->
                <div class="prediction-hero pulse-animation" id="predictionHero">
                    <div class="prediction-title"><i class="fa-solid fa-award"></i> Most Likely Allocated Course</div>
                    <div class="prediction-value" id="predictedBranch">Awaiting Input Submission...</div>
                </div>

                <!-- PROBABILITY BREAKDOWN -->
                <div class="analytics-section">
                    <h3><i class="fa-solid fa-network-wire" style="color:var(--secondary)"></i> Allocation Probability Top Matches</h3>
                    <div id="probContainer">
                        <p style="color:var(--text-muted); font-size: 0.9rem; text-align:center; padding: 25px 0;">
                            Submit candidate parameters to render likelihood charts.
                        </p>
                    </div>
                </div>
            </div>

            <!-- METRICS SUMMARY -->
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-icon"><i class="fa-solid fa-microchip"></i></div>
                    <div class="metric-info">
                        <label>ML Engine</label>
                        <span>XGBoost Multi-Class</span>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon" style="color:var(--accent); background:rgba(6,182,212,0.15)"><i class="fa-solid fa-shield-halved"></i></div>
                    <div class="metric-info">
                        <label>Host Environment</label>
                        <span>Hugging Face Docker</span>
                    </div>
                </div>
            </div>

        </div>

    </div>

    <!-- PREDICTION HISTORY CARD -->
    <div class="glass-card history-card" id="historySection" style="display: none;">
        <div class="card-header">
            <h2><i class="fa-solid fa-clock-rotate-left" style="color:var(--primary)"></i> Session Prediction History</h2>
            <p>Recent branch allocation inferences run in this session</p>
        </div>
        <table class="history-table">
            <thead>
                <tr>
                    <th>Institute</th>
                    <th>Percentile</th>
                    <th>Category</th>
                    <th>Quota</th>
                    <th>Predicted Course</th>
                </tr>
            </thead>
            <tbody id="historyBody">
            </tbody>
        </table>
    </div>

    <!-- FOOTER -->
    <footer>
        &copy; 2026 AI Admission Allocation Engine. Deployed on Hugging Face Spaces.
    </footer>

    <!-- INTERACTIVE SCRIPT -->
    <script>
        const predictionHistory = [];

        function setTheme(themeName, btnElement) {
            document.documentElement.setAttribute('data-theme', themeName);
            document.querySelectorAll('.theme-btn').forEach(btn => btn.classList.remove('active'));
            btnElement.classList.add('active');
        }

        async function handlePredict(e) {
            e.preventDefault();

            const btn = document.getElementById('submitBtn');
            btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing Inference...`;
            btn.disabled = true;

            const payload = {
                gender: document.getElementById('gender').value,
                category: document.getElementById('category').value,
                percentile: parseFloat(document.getElementById('percentile').value),
                seat: document.getElementById('seat').value,
                institute: document.getElementById('institute').value
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const res = await response.json();

                if (res.status === 'success') {
                    document.getElementById('predictedBranch').innerText = res.prediction;

                    const probContainer = document.getElementById('probContainer');
                    probContainer.innerHTML = '';

                    if (res.top_branches.length === 0) {
                        probContainer.innerHTML = `<p style="color:var(--text-muted); font-size: 0.9rem; text-align:center;">No high-confidence branches predicted for this percentile.</p>`;
                    } else {
                        res.top_branches.forEach(item => {
                            const probItem = document.createElement('div');
                            probItem.className = 'prob-item';
                            probItem.innerHTML = `
                                <div class="prob-header">
                                    <span class="prob-name" title="${item.branch}">${item.branch}</span>
                                    <span class="prob-val">${item.prob}%</span>
                                </div>
                                <div class="progress-bg">
                                    <div class="progress-fill" style="width: 0%"></div>
                                </div>
                            `;
                            probContainer.appendChild(probItem);

                            setTimeout(() => {
                                probItem.querySelector('.progress-fill').style.width = item.prob + '%';
                            }, 100);
                        });
                    }

                    predictionHistory.unshift({
                        institute: payload.institute,
                        percentile: payload.percentile,
                        category: payload.category,
                        seat: payload.seat,
                        prediction: res.prediction
                    });
                    renderHistory();

                } else {
                    alert('Prediction Error: ' + res.message);
                }
            } catch (err) {
                alert('Server Connection Failed or Invalid Input Data.');
            } finally {
                btn.innerHTML = `<span>Execute Inference</span><i class="fa-solid fa-wand-magic-sparkles"></i>`;
                btn.disabled = false;
            }
        }

        function renderHistory() {
            const section = document.getElementById('historySection');
            const tbody = document.getElementById('historyBody');
            section.style.display = 'block';
            tbody.innerHTML = '';

            predictionHistory.slice(0, 5).forEach(item => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${item.institute}</td>
                    <td><strong>${item.percentile}%ile</strong></td>
                    <td>${item.category}</td>
                    <td>${item.seat}</td>
                    <td style="color:var(--accent); font-weight:700;">${item.prediction}</td>
                `;
                tbody.appendChild(tr);
            });
        }
    </script>
</body>
</html>
"""

# ==============================================================================
# 4. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
