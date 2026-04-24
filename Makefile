# ── Emotion Intelligence Platform — Task Runner ───────────────────

.PHONY: install install-platform preprocess test test-unit test-integration \
        test-all serve serve-dev train-baseline loso-all tune-grid-all \
        tune-optuna-all docker-build docker-up docker-down lint clean

# ── Setup ─────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

install-platform:
	pip install -r requirements_platform.txt

# ── Data ──────────────────────────────────────────────────────────

preprocess:
	python src/data/save_processed.py --config configs/default.yaml

preprocess-overwrite:
	python src/data/save_processed.py --config configs/default.yaml --overwrite

# ── Testing ───────────────────────────────────────────────────────

test-unit:
	python -m pytest tests/test_loader.py tests/test_models.py \
	                 tests/test_tuner.py tests/test_inference.py -v --tb=short

test-integration:
	python -m pytest tests/integration/test_platform.py -v --tb=short

test-all: test-unit test-integration

lint:
	python -m py_compile \
	  src/schemas/models.py src/storage/database.py \
	  src/pipeline/signal_pipeline.py src/insights/engine.py \
	  src/api/routes.py src/data/loader.py src/data/preprocessor.py \
	  src/data/save_processed.py src/data/dataset.py \
	  src/data/cached_dataset.py src/data/splits.py \
	  src/features/eeg_features.py src/features/ecg_features.py \
	  src/models/baseline.py src/models/deep_model.py \
	  src/training/trainer.py src/training/evaluator.py \
	  src/training/cross_subject_eval.py src/training/tuner.py \
	  src/inference/predict.py src/utils/config.py src/utils/logger.py
	@echo "✅ All files syntax-clean"

# ── API Server ────────────────────────────────────────────────────

serve:
	uvicorn src.api.routes:app --host 0.0.0.0 --port 8000

serve-dev:
	uvicorn src.api.routes:app --host 0.0.0.0 --port 8000 --reload

# Start with trained models (set paths as needed)
serve-with-models:
	python -c "\
from src.api.routes import create_app; import uvicorn; \
app = create_app(model_paths={ \
    'valence'  : 'outputs/models/best_valence_FusionModel.pt', \
    'arousal'  : 'outputs/models/best_arousal_FusionModel.pt', \
    'dominance': 'outputs/models/best_dominance_FusionModel.pt'}); \
uvicorn.run(app, host='0.0.0.0', port=8000)"

# ── Training (run on Colab, listed here for reference) ────────────

train-baseline:
	@echo "Open notebooks/experiments/03_baseline_colab.ipynb on Colab"

loso-valence:
	python src/training/cross_subject_eval.py --target valence --model_type fusion

loso-arousal:
	python src/training/cross_subject_eval.py --target arousal --model_type fusion

loso-dominance:
	python src/training/cross_subject_eval.py --target dominance --model_type fusion

loso-all: loso-valence loso-arousal loso-dominance

tune-grid-all:
	for target in valence arousal dominance; do \
	  for model in rf svm logreg gbm; do \
	    python src/training/tuner.py --mode baseline \
	      --model_type $$model --target $$target; \
	  done; \
	done

tune-optuna-all:
	for target in valence arousal dominance; do \
	  python src/training/tuner.py --mode deep \
	    --target $$target --n_trials 30; \
	done

# ── Inference ─────────────────────────────────────────────────────

predict-sample:
	python src/inference/predict.py \
	  --eeg_path  data/raw/sample_eeg.npy \
	  --ecg_path  data/raw/sample_ecg.npy \
	  --model_path outputs/models/best_valence_FusionModel.pt \
	  --model_type fusion --target valence \
	  --out outputs/results/sample_prediction.json

# ── Docker ────────────────────────────────────────────────────────

docker-build:
	docker build -t emotion-platform .

docker-up:
	docker-compose up --build -d
	@echo "API running at http://localhost:8000/docs"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f api

# ── Export ────────────────────────────────────────────────────────

export-db:
	python -c "\
from src.storage.database import PredictionDB; \
db = PredictionDB(); \
db.export_parquet('outputs/platform/predictions.parquet'); \
print('Exported')"

# ── Cleanup ───────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.log" -path "*/outputs/*" -delete 2>/dev/null || true
	@echo "✅ Cleaned"

clean-platform:
	rm -f outputs/platform/predictions.db
	rm -f outputs/platform/predictions.parquet
	rm -f outputs/platform/predictions.csv
	@echo "✅ Platform data cleared"