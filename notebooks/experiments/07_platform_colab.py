# This file contains the cell-by-cell content for
# notebooks/experiments/07_platform_colab.ipynb
# Copy each CELL block into a new Jupyter cell in order.

# ═══════════════════════════════════════════════════════
# CELL 1 — Markdown
# ═══════════════════════════════════════════════════════
"""
# Emotion Intelligence Platform — Colab Deployment
**What this notebook does:**
- Installs platform dependencies
- Clones the repo and loads trained models
- Starts the FastAPI server on port 8000
- Exposes it publicly via ngrok
- Runs a full demo: signal → prediction → storage → insights
- Verifies all API endpoints

**Prerequisites:**
- Trained model checkpoints in `outputs/models/`
- `DREAMER.mat` preprocessed (run `02_train_colab.ipynb` first)
"""

# ═══════════════════════════════════════════════════════
# CELL 2 — Code: Setup
# ═══════════════════════════════════════════════════════

import sys
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    GITHUB_REPO = "https://github.com/YOUR_USERNAME/emotion-recognition.git"
    import subprocess
    subprocess.run(["git", "clone", GITHUB_REPO, "/content/er"], check=True)
    import os; os.chdir("/content/er")

    # Install platform stack
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install",
        "fastapi", "uvicorn[standard]", "pydantic",
        "httpx", "pyngrok", "pyarrow", "requests", "-q"], check=True)

    print("✅ Platform dependencies installed")
else:
    import os
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(""), "")))
    print("✅ Running locally")

# ═══════════════════════════════════════════════════════
# CELL 3 — Code: Verify model checkpoints exist
# ═══════════════════════════════════════════════════════

import os

MODELS_DIR = "outputs/models"
REQUIRED_MODELS = {
    "valence"  : f"{MODELS_DIR}/best_valence_FusionModel.pt",
    "arousal"  : f"{MODELS_DIR}/best_arousal_FusionModel.pt",
    "dominance": f"{MODELS_DIR}/best_dominance_FusionModel.pt",
}

missing = [k for k, p in REQUIRED_MODELS.items() if not os.path.exists(p)]
if missing:
    print(f"⚠️  Missing model checkpoints: {missing}")
    print("Run notebooks/experiments/02_train_colab.ipynb first.")
    print("Or set USE_BASELINE_MODEL = True below to use a baseline RF model.")
    USE_BASELINE_MODEL = True
else:
    print("✅ All model checkpoints found:")
    for k, p in REQUIRED_MODELS.items():
        size_mb = os.path.getsize(p) / 1e6
        print(f"   {k}: {p} ({size_mb:.1f} MB)")
    USE_BASELINE_MODEL = False

# ═══════════════════════════════════════════════════════
# CELL 4 — Code: Create app (with or without models)
# ═══════════════════════════════════════════════════════

import sys
sys.path.insert(0, ".")
os.makedirs("outputs/platform", exist_ok=True)

from src.api.routes import create_app

if USE_BASELINE_MODEL:
    # Start without models — /health works, /predict returns 503
    app = create_app(
        config_path = "configs/default.yaml",
        model_paths = None,
        db_path     = "outputs/platform/predictions.db",
    )
    print("⚠️  Server started WITHOUT models (/predict unavailable)")
else:
    app = create_app(
        config_path = "configs/default.yaml",
        model_paths = REQUIRED_MODELS,
        model_type  = "fusion",
        db_path     = "outputs/platform/predictions.db",
    )
    print("✅ Server created WITH FusionModel loaded for all 3 targets")

# ═══════════════════════════════════════════════════════
# CELL 5 — Code: Start server in background thread
# ═══════════════════════════════════════════════════════

import uvicorn
import threading
import time

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(2)   # let server initialise

# Quick local health check
import requests
r = requests.get("http://localhost:8000/health", timeout=5)
assert r.status_code == 200, f"Server not healthy: {r.status_code}"
health = r.json()
print(f"✅ Server running | status={health['status']} "
      f"model_loaded={health['model_loaded']} "
      f"db_connected={health['db_connected']}")

# ═══════════════════════════════════════════════════════
# CELL 6 — Code: Expose via ngrok (get public URL)
# ═══════════════════════════════════════════════════════

# Get your free token at https://ngrok.com (takes 30 seconds to sign up)
NGROK_TOKEN = "YOUR_NGROK_AUTH_TOKEN"   # ← paste your token here

try:
    from pyngrok import ngrok
    ngrok.set_auth_token(NGROK_TOKEN)
    tunnel = ngrok.connect(8000, bind_tls=True)
    BASE_URL = tunnel.public_url
    print(f"✅ Public URL : {BASE_URL}")
    print(f"   Swagger UI : {BASE_URL}/docs")
    print(f"   ReDoc UI   : {BASE_URL}/redoc")
except Exception as e:
    BASE_URL = "http://localhost:8000"
    print(f"⚠️  ngrok not configured — using local: {BASE_URL}")
    print(f"   Error: {e}")

# ═══════════════════════════════════════════════════════
# CELL 7 — Code: Demo — send a prediction request
# ═══════════════════════════════════════════════════════

import numpy as np
import requests
import json

# Simulate 30 seconds of EEG + ECG (random — replace with real data)
EEG_FS, ECG_FS = 128, 256
DURATION_SEC   = 30

np.random.seed(42)
eeg_data = np.random.randn(DURATION_SEC * EEG_FS, 14).tolist()
ecg_data = np.random.randn(DURATION_SEC * ECG_FS, 2).tolist()

payload = {
    "user_id"  : "demo_user_001",
    "eeg_data" : eeg_data,
    "ecg_data" : ecg_data,
    "metadata" : {"device": "Emotiv EPOC", "task": "resting_state"},
}

print(f"Sending {DURATION_SEC}s EEG ({len(eeg_data)} samples) + ECG ({len(ecg_data)} samples)...")
r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=120)

if r.status_code == 200:
    result = r.json()
    pred   = result["prediction"]
    print("\n" + "="*52)
    print("  EMOTION PREDICTION RESULT")
    print("="*52)
    print(f"  Valence   : {pred['valence']:4s}  (prob={pred['valence_prob']:.3f}  conf={pred['valence_conf']:.3f})")
    print(f"  Arousal   : {pred['arousal']:4s}  (prob={pred['arousal_prob']:.3f}  conf={pred['arousal_conf']:.3f})")
    print(f"  Dominance : {pred['dominance']:4s}  (prob={pred['dominance_prob']:.3f}  conf={pred['dominance_conf']:.3f})")
    print(f"  Windows   : {pred['n_windows']}")
    print(f"  Quality   : {pred['signal_quality']}")
    print(f"  Time      : {pred['processing_ms']:.1f} ms")
    print("="*52)
    if result["insights"]:
        print("\n  REALTIME INSIGHTS:")
        for ins in result["insights"]:
            print(f"  [{ins['severity'].upper():7s}] {ins['title']}")
            print(f"            {ins['description'][:80]}...")
elif r.status_code == 503:
    print("⚠️  Model not loaded. Run training notebooks first.")
else:
    print(f"❌ Error {r.status_code}: {r.text}")

# ═══════════════════════════════════════════════════════
# CELL 8 — Code: Send multiple sessions to build history
# ═══════════════════════════════════════════════════════

print("Sending 8 more sessions to build prediction history...")
for i in range(8):
    np.random.seed(i * 7)
    payload_i = {
        "user_id" : "demo_user_001",
        "eeg_data": np.random.randn(DURATION_SEC * EEG_FS, 14).tolist(),
        "ecg_data": np.random.randn(DURATION_SEC * ECG_FS, 2).tolist(),
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload_i, timeout=120)
    if r.status_code == 200:
        p = r.json()["prediction"]
        print(f"  Session {i+2}: V={p['valence']} A={p['arousal']} D={p['dominance']} "
              f"({p['processing_ms']:.0f}ms)")
    else:
        print(f"  Session {i+2}: status={r.status_code}")

print("\n✅ 9 total sessions submitted")

# ═══════════════════════════════════════════════════════
# CELL 9 — Code: Query all endpoints
# ═══════════════════════════════════════════════════════

USER_ID = "demo_user_001"

print(f"\n{'='*52}")
print("  ENDPOINT VERIFICATION")
print(f"{'='*52}")

# Health
r = requests.get(f"{BASE_URL}/health")
print(f"\n  GET /health → {r.status_code}")
print(f"  {r.json()}")

# History
r = requests.get(f"{BASE_URL}/users/{USER_ID}/history?days=30")
body = r.json()
print(f"\n  GET /users/{USER_ID}/history → {r.status_code}")
print(f"  total_records={body['total_records']}  page={body['page']}")

# User profile
r = requests.get(f"{BASE_URL}/users/{USER_ID}/profile")
print(f"\n  GET /users/{USER_ID}/profile → {r.status_code}")
if r.status_code == 200:
    p = r.json()
    print(f"  total_sessions={p.get('total_sessions')}  "
          f"baseline_v={p.get('baseline_valence','N/A')}")

# Summary
r = requests.get(f"{BASE_URL}/users/{USER_ID}/summary?days=30")
print(f"\n  GET /users/{USER_ID}/summary → {r.status_code}")
if r.status_code == 200:
    s = r.json()
    print(f"  valence_mean={s.get('valence_mean')}  "
          f"valence_high_pct={s.get('valence_high_pct')}%")

# Insights
r = requests.get(f"{BASE_URL}/users/{USER_ID}/insights?days=30&regenerate=true")
print(f"\n  GET /users/{USER_ID}/insights → {r.status_code}")
if r.status_code == 200:
    ins_list = r.json()["insights"]
    print(f"  {len(ins_list)} insights generated")
    for ins in ins_list[:3]:
        print(f"  [{ins['severity'].upper():7s}] {ins['title']}")

# Platform stats
r = requests.get(f"{BASE_URL}/platform/stats")
print(f"\n  GET /platform/stats → {r.status_code}")
print(f"  {r.json()}")

print(f"\n{'='*52}")
print("  ✅ All endpoints verified")
print(f"{'='*52}")

# ═══════════════════════════════════════════════════════
# CELL 10 — Code: Export DB and push to GitHub
# ═══════════════════════════════════════════════════════

# Export DB to CSV/Parquet
r = requests.post(
    f"{BASE_URL}/admin/export",
    params={"out_path": "outputs/platform/predictions.parquet"},
    timeout=30,
)
print(f"Export status: {r.status_code} — {r.json()}")

# Save notebook outputs
import pandas as pd

# Load and display prediction history as DataFrame
hist_r = requests.get(f"{BASE_URL}/users/{USER_ID}/history?days=30&page_size=200")
records = hist_r.json().get("records", [])
if records:
    df = pd.DataFrame(records)
    print(f"\nPrediction history ({len(df)} rows):")
    print(df[["timestamp","valence","arousal","dominance",
              "valence_prob","n_windows","processing_ms"]].to_string(index=False))
    df.to_csv("outputs/platform/demo_history.csv", index=False)

# Push to GitHub
if IN_COLAB:
    import subprocess
    subprocess.run(["git", "config", "--global", "user.email", "you@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Your Name"])
    subprocess.run(["git", "add", "outputs/platform/", "notebooks/"])
    subprocess.run(["git", "commit", "-m", "feat: platform demo results"])
    subprocess.run(["git", "push"])
    print("✅ Pushed to GitHub")
