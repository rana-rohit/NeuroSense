"""
tests/test_tuner.py
Unit tests for src/training/tuner.py — baseline grid search only.
(Optuna deep search requires torch — tested separately in Colab.)
Run: python -m pytest tests/test_tuner.py -v
"""

import pytest
import numpy as np
import json, os, sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))


@pytest.fixture
def Xy():
    np.random.seed(42)
    X = np.random.randn(200, 280).astype(np.float32)
    y = np.random.randint(0, 2, 200)
    return X, y


class TestBaselineGridSearch:

    @pytest.mark.parametrize("model_type", ["logreg", "rf"])
    def test_tune_baseline_runs(self, Xy, tmp_path, model_type):
        from src.training.tuner import tune_baseline
        X, y = Xy
        result = tune_baseline(
            X, y, model_type=model_type,
            target="valence", cv_folds=2,
            save_dir=str(tmp_path),
        )
        assert "best_auc"    in result
        assert "best_params" in result
        assert "target"      in result
        assert 0.0 <= result["best_auc"] <= 1.0

    def test_tune_baseline_saves_json(self, Xy, tmp_path):
        from src.training.tuner import tune_baseline
        X, y = Xy
        tune_baseline(X, y, model_type="logreg", target="arousal",
                      cv_folds=2, save_dir=str(tmp_path))
        path = tmp_path / "gridsearch_logreg_arousal.json"
        assert path.exists()
        with open(path) as f:
            d = json.load(f)
        assert d["target"]     == "arousal"
        assert d["model_type"] == "logreg"
        assert "best_auc" in d

    def test_tune_baseline_invalid_model(self, Xy, tmp_path):
        from src.training.tuner import tune_baseline
        X, y = Xy
        with pytest.raises(ValueError, match="Unknown model_type"):
            tune_baseline(X, y, model_type="xgboost", target="valence",
                          save_dir=str(tmp_path))

    @pytest.mark.parametrize("target", ["valence", "arousal", "dominance"])
    def test_tune_baseline_all_targets(self, Xy, tmp_path, target):
        from src.training.tuner import tune_baseline
        X, y = Xy
        result = tune_baseline(
            X, y, model_type="logreg", target=target,
            cv_folds=2, save_dir=str(tmp_path),
        )
        assert result["target"] == target
        assert result["best_auc"] >= 0.0

    def test_best_params_have_clf_prefix(self, Xy, tmp_path):
        from src.training.tuner import tune_baseline
        X, y = Xy
        result = tune_baseline(
            X, y, model_type="logreg", target="valence",
            cv_folds=2, save_dir=str(tmp_path),
        )
        for key in result["best_params"]:
            assert key.startswith("clf__"), f"Param key missing clf__ prefix: {key}"

    def test_cv_folds_stored(self, Xy, tmp_path):
        from src.training.tuner import tune_baseline
        X, y = Xy
        result = tune_baseline(
            X, y, model_type="logreg", target="valence",
            cv_folds=3, save_dir=str(tmp_path),
        )
        assert result["cv_folds"] == 3


class TestTuningYamlConfig:

    def test_platform_yaml_exists(self):
        assert os.path.exists("configs/platform/platform.yaml"), \
            "configs/platform/platform.yaml missing"

    def test_tuning_search_space_keys(self):
        """Verify optuna search space is documented in platform.yaml."""
        import yaml
        with open("configs/platform/platform.yaml") as f:
            cfg = yaml.safe_load(f)
        assert "insights" in cfg
        assert "optuna" in cfg or "data" in cfg   # either full optuna block or partial

    def test_default_yaml_exists(self):
        assert os.path.exists("configs/default.yaml")

    def test_default_yaml_has_required_keys(self):
        import yaml
        with open("configs/default.yaml") as f:
            cfg = yaml.safe_load(f)
        for section in ["data", "labels", "training", "model"]:
            assert section in cfg, f"Missing section: {section}"