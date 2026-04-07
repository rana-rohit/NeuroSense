# Emotion Recognition System
### EEG + ECG · DREAMER Dataset · PyTorch · Scikit-learn

> Predicts human emotional states (Valence, Arousal, Dominance) from
> physiological signals using hand-crafted features and deep learning.

---

## Project Structure

```text
emotion-recognition/
├── data/
│   ├── raw/                  # DREAMER.mat (not tracked in git)
│   ├── processed/            # Preprocessed .npy cache + labels.csv
│   └── splits/               # Saved train/val/test index arrays
│
├── notebooks/
│   ├── exploration/
│   │   └── 01_data_exploration.ipynb
│   └── experiments/
│       ├── 02_train_colab.ipynb
│       ├── 03_baseline_colab.ipynb
│       └── 04_deep_loso_colab.ipynb
│
├── src/
│   ├── data/
│   │   ├── loader.py          # Raw .mat loading + trial accessors
│   │   ├── preprocessor.py    # Filter, baseline, normalise, segment
│   │   ├── dataset.py         # PyTorch Dataset (from raw)
│   │   ├── cached_dataset.py  # PyTorch Dataset (from .npy cache)
│   │   ├── save_processed.py  # One-time preprocessing + cache script
│   │   └── splits.py          # Train/val/test + LOSO split helpers
│   ├── features/
│   │   ├── eeg_features.py    # 258-dim EEG feature vector
│   │   └── ecg_features.py    # 22-dim ECG + HRV feature vector
│   ├── models/
│   │   ├── baseline.py        # LogReg / SVM / RF / GBM pipelines
│   │   └── deep_model.py      # EEGNet, CNN1D, CNNLSTM, FusionModel
│   ├── training/
│   │   ├── trainer.py         # Training loop + early stopping
│   │   ├── evaluator.py       # Metrics, confusion matrix, ROC/PR curves
│   │   └── cross_subject_eval.py  # Deep LOSO engine + CLI
│   └── utils/
│       ├── config.py          # YAML config loader
│       └── logger.py          # Console + file logger
│
├── configs/
│   └── default.yaml
├── outputs/
│   ├── models/                # Saved .pt / .pkl checkpoints
│   ├── logs/                  # Training logs
│   └── results/               # Metrics, plots, CSVs
├── tests/
│   ├── test_loader.py
│   └── test_models.py
├── .gitignore
├── requirements.txt
└── README.md
```
---

## Quick Start

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/emotion-recognition.git
cd emotion-recognition
pip install -r requirements.txt
```

### 2. Add the dataset
Place `DREAMER.mat` in `data/raw/DREAMER.mat`.  
_(Dataset must be requested from the original authors.)_

### 3. Preprocess and cache (run once)
```bash
python src/data/save_processed.py --config configs/default.yaml
```

### 4. Run tests
```bash
python -m pytest tests/ -v
```

### 5. Train on Colab
1. Push repo to GitHub  
2. Upload `DREAMER.mat` to Google Drive  
3. Open `notebooks/experiments/02_train_colab.ipynb` in Colab  
4. Run all cells

---

## Dataset

| Property | Value |
|----------|-------|
| Subjects | 23 |
| Stimuli  | 18 film clips per subject |
| EEG      | 14 channels · 128 Hz |
| ECG      | 2 channels · 256 Hz |
| Labels   | Valence, Arousal, Dominance (1–5, binarised at 3) |

---

## Models

| Model | Type | Input |
|-------|------|-------|
| LogReg / SVM / RF / GBM | Classical ML | 280 hand-crafted features |
| EEGNet | Compact CNN | EEG only |
| FusionModel (CNN) | Late-fusion CNN | EEG + ECG |
| FusionModel (CNNLSTM) | CNN + BiLSTM | EEG + ECG |

---

## Evaluation Protocol

- **Within-subject** — 80/20 split per subject (upper bound)  
- **Cross-subject LOSO** — Leave-One-Subject-Out (23 folds, true generalisation)

---

## Configuration

All hyperparameters are in `configs/default.yaml`.  
Key settings:
```yaml
data:
  segment_length: 4      # window size (seconds)
  overlap: 2             # overlap (seconds)
  norm_method: zscore

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  patience: 10

model:
  type: fusion           # eegnet | cnn | cnnlstm | fusion
```

---

## References

- Katsigiannis & Ramzan (2018). _DREAMER: A Database for Emotion Recognition
  Through EEG and ECG Signals._ IEEE JBHI.  
- Lawhern et al. (2018). _EEGNet: A Compact CNN for EEG-based BCIs._ J. Neural Eng.