# Emotion Intelligence Platform
### EEG + ECG · DREAMER Dataset · PyTorch · FastAPI · SQLite

> Converts raw physiological signals (EEG + ECG) into persistent, trackable,
> and actionable emotional insights via a production-grade REST API.

---

## What This Is

This project started as an ML experiment and has evolved into a full backend
intelligence platform. The model is one component within a larger system that:

- Accepts EEG + ECG signals via REST API
- Preprocesses, segments, and runs multi-target inference
- Stores every prediction in SQLite with full history
- Generates behavioural insights (trends, anomalies, peaks, correlations)
- Tracks per-user emotional baselines over time

---

## System Flow

```
Signal Input → Quality Check → Preprocess → Inference → Aggregate
     → Store (SQLite) → Generate Insights → API Response
```

---

## Project Structure

```
emotion-recognition/
├── src/
│   ├── schemas/          # Pydantic data contracts
│   ├── pipeline/         # Signal orchestration
│   ├── storage/          # SQLite database layer
│   ├── insights/         # Insight engine (5 detectors)
│   ├── api/              # FastAPI REST endpoints
│   ├── data/             # DREAMER loader, preprocessor, datasets
│   ├── features/         # EEG (258-dim) + ECG (22-dim) features
│   ├── models/           # Baseline ML + deep learning models
│   ├── training/         # Trainer, evaluator, LOSO, tuner
│   ├── inference/        # Standalone predictor
│   └── utils/            # Config, logger
├── notebooks/
│   ├── exploration/      # 01_data_exploration.ipynb
│   └── experiments/      # 02–07 training + platform notebooks
├── configs/
│   ├── default.yaml      # ML hyperparameters
│   └── platform/         # Platform config
├── tests/
│   ├── test_loader.py
│   ├── test_models.py
│   ├── test_tuner.py
│   ├── test_inference.py
│   └── integration/      # Full platform integration tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt          # ML dependencies
└── requirements_platform.txt # Full platform stack
```

---

## Quick Start

### 1. Install
```bash
git clone https://github.com/YOUR/emotion-recognition.git
cd emotion-recognition
pip install -r requirements_platform.txt
```

### 2. Preprocess data (run once)
```bash
python src/data/save_processed.py --config configs/default.yaml
```

### 3. Train models
Open `notebooks/experiments/02_train_colab.ipynb` on Google Colab.

### 4. Start the platform API
```bash
# Without models (health endpoint only)
uvicorn src.api.routes:app --reload --port 8000

# With models
python -c "
from src.api.routes import create_app
import uvicorn
app = create_app(model_paths={
    'valence'  : 'outputs/models/best_valence_FusionModel.pt',
    'arousal'  : 'outputs/models/best_arousal_FusionModel.pt',
    'dominance': 'outputs/models/best_dominance_FusionModel.pt',
})
uvicorn.run(app, host='0.0.0.0', port=8000)
"
```

### 5. Open Swagger UI
```
http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Platform liveness + readiness |
| POST | `/predict` | Submit EEG/ECG → prediction + insights |
| GET | `/users/{id}/history` | Paginated prediction history |
| GET | `/users/{id}/insights` | Generated behavioural insights |
| GET | `/users/{id}/profile` | User emotional baseline profile |
| GET | `/users/{id}/summary` | Aggregated stats by dimension |
| GET | `/platform/stats` | Admin: total predictions/users/sessions |
| POST | `/admin/export` | Export DB to Parquet or CSV |

---

## Prediction Request

```json
{
  "user_id": "patient_001",
  "eeg_data": [[ch0..ch13], ...],
  "ecg_data": [[ch0, ch1], ...],
  "eeg_fs": 128.0,
  "ecg_fs": 256.0
}
```

Minimum signal: 5 seconds. EEG shape `(N, 14)`. ECG shape `(M, 2)`.

---

## Insight Types

| Type | Trigger | Example |
|------|---------|---------|
| TREND | Rolling mean shift > 0.12 over 5 sessions | "Valence trending upward (+0.18)" |
| ANOMALY | Z-score > 2.0 from personal mean | "Unusual arousal spike (2.8 SD)" |
| PEAK | New all-time high/low | "Personal best valence: 0.82" |
| STABILITY | CV < 0.10 over 7 days | "Stable High valence this week" |
| CORRELATION | Pearson r > 0.65 | "Excited pattern: valence + arousal co-vary" |

---

## Docker

```bash
docker-compose up
# API at http://localhost:8000/docs
```

---

## Running Tests

```bash
# Unit tests
pytest tests/ -v

# Integration tests (no models needed)
pytest tests/integration/test_platform.py -v

# All tests
make test-all
```

---

## Dataset

DREAMER — 23 subjects, 18 film clips, EEG (14ch @ 128Hz) + ECG (2ch @ 256Hz).
Labels: Valence, Arousal, Dominance (1–5, binarised at threshold 3).
Request access: [Zenodo](https://zenodo.org/records/546113)

---

## Models

| Model | Type | Input | LOSO AUC |
|-------|------|-------|----------|
| Random Forest | Classical ML | 280 features | ~0.65-0.70 |
| FusionModel (CNNLSTM) | Deep | EEG + ECG | ~0.68-0.75 |
| Tuned FusionModel | Deep + Optuna | EEG + ECG | ~0.72-0.78 |

---

## References

- Katsigiannis & Ramzan (2018). *DREAMER.* IEEE JBHI.
- Lawhern et al. (2018). *EEGNet.* J. Neural Eng.