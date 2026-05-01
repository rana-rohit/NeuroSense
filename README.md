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

### ✨ Recent Evaluation & Pipeline Upgrades
The ML pipeline has been rigorously upgraded to ensure strict, leakage-free evaluation:
- **Leakage-Free LOSO**: Full 23-subject Leave-One-Subject-Out evaluation. Flip decisions are strictly based on validation predictions.
- **Subject-Level Validation**: Deterministic but randomized validation split ensuring no subject-level or window-level overlap between train, val, and test sets.
- **Video-Level Aggregation**: Metrics are computed on video-level aggregated predictions (mean probability) rather than isolated, overlapping windows, ensuring realistic performance reporting.
- **Baseline-Aware Preprocessing**: Improved single-pass normalization preserving crucial signal amplitude information instead of destroying it via double z-scoring.
- **Advanced Feature Engineering**: Mathematically sound feature pipelines using linear bandpower for ratios and log-bandpower for absolute features and asymmetry (now including the O1-O2 pair).
- **Modality Ablation**: Built-in CLI support for training `eeg` only, `ecg` only, or `fusion` models.

---

## System Flow

```text
Signal Input → Quality Check → Baseline-Aware Preprocess → Inference
     → Aggregate → Store (SQLite) → Generate Insights → API Response
```

---

## Project Structure

```text
emotion-recognition/
├── src/
│   ├── schemas/          # Pydantic data contracts
│   ├── pipeline/         # Signal orchestration
│   ├── storage/          # SQLite database layer
│   ├── insights/         # Insight engine (5 detectors)
│   ├── api/              # FastAPI REST endpoints
│   ├── data/             # DREAMER loader, preprocessor, datasets
│   ├── features/         # EEG (258-dim) + ECG (22-dim) features
│   ├── models/           # Baseline ML + deep learning models (EEGNet, Fusion)
│   ├── training/         # Trainer, evaluator, LOSO loop, tuner
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

### 2. Evaluate Models (Full LOSO)
Run the fully calibrated, leakage-free LOSO evaluation pipeline across all 23 subjects:
```bash
python src/training/cross_subject_eval.py \
    --target valence \
    --model_type fusion \
    --modality fusion \
    --config configs/default.yaml
```
*(You can test modality ablation by swapping `--modality` with `eeg` or `ecg`)*

### 3. Start the Platform API
```bash
# Without models (health endpoint only)
uvicorn src.api.routes:app --reload --port 8000

# With models (Requires pre-trained weights)
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

### 4. Open Swagger UI
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

## Dataset

DREAMER — 23 subjects, 18 film clips, EEG (14ch @ 128Hz) + ECG (2ch @ 256Hz).
Labels: Valence, Arousal, Dominance (1–5, binarised at threshold 3).
Request access: [Zenodo](https://zenodo.org/records/546113)

---

## Models

The system evaluates models using robust Leave-One-Subject-Out (LOSO) cross-validation with video-level metric aggregation and disjoint, subject-level train/validation sets. Double-balancing artifacts have been removed, relying purely on weighted loss for imbalanced datasets.

| Model | Type | Input | Feature Dimension |
|-------|------|-------|-------------------|
| Random Forest | Classical ML | EEG + ECG | ~280 |
| FusionModel (CNNLSTM) | Deep Late-Fusion | EEG + ECG | Temporal CNN+LSTM |
| EEGNet | Deep Spatial-Temporal | EEG | Spatial/Temporal Convs |

---

## References

- Katsigiannis & Ramzan (2018). *DREAMER.* IEEE JBHI.
- Lawhern et al. (2018). *EEGNet.* J. Neural Eng.