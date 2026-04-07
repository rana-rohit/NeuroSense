# ── Emotion Recognition — Task Runner ────────────────────────────

.PHONY: install preprocess test train-baseline train-deep lint clean

install:
	pip install -r requirements.txt

preprocess:
	python src/data/save_processed.py --config configs/default.yaml

test:# ── Emotion Recognition — Task Runner ────────────────────────────

.PHONY: install preprocess test train-baseline train-deep lint clean

install:
	pip install -r requirements.txt

preprocess:
	python src/data/save_processed.py --config configs/default.yaml

test:
	python -m pytest tests/ -v --tb=short

train-baseline:
	# Run baseline notebook via nbconvert on Colab
	@echo "Open notebooks/experiments/03_baseline_colab.ipynb on Colab"

loso-valence:
	python src/training/cross_subject_eval.py \
		--target valence --model_type fusion

loso-arousal:
	python src/training/cross_subject_eval.py \
		--target arousal --model_type fusion

loso-dominance:
	python src/training/cross_subject_eval.py \
		--target dominance --model_type fusion

loso-all: loso-valence loso-arousal loso-dominance

lint:
	python -m py_compile src/data/*.py src/features/*.py \
		src/models/*.py src/training/*.py src/utils/*.py
	@echo "✅ No syntax errors"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete; \
	@echo "✅ Cleaned"
	python -m pytest tests/ -v --tb=short

train-baseline:
	# Run baseline notebook via nbconvert on Colab
	@echo "Open notebooks/experiments/03_baseline_colab.ipynb on Colab"

loso-valence:
	python src/training/cross_subject_eval.py \
		--target valence --model_type fusion

loso-arousal:
	python src/training/cross_subject_eval.py \
		--target arousal --model_type fusion

loso-dominance:
	python src/training/cross_subject_eval.py \
		--target dominance --model_type fusion

loso-all: loso-valence loso-arousal loso-dominance

lint:
	python -m py_compile src/data/*.py src/features/*.py \
		src/models/*.py src/training/*.py src/utils/*.py
	@echo "✅ No syntax errors"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete; \
	@echo "✅ Cleaned"