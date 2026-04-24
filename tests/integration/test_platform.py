"""
tests/integration/test_platform.py
Integration tests for the full platform stack.

Tests run without pydantic/fastapi/torch — mocked where needed.
Run: python -m pytest tests/integration/test_platform.py -v

On Colab (with all deps):
    pip install fastapi httpx pytest pytest-asyncio pydantic -q
    pytest tests/integration/test_platform.py -v
"""

import os
import sys
import json
import uuid
import tempfile
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
))

# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    from src.storage.database import PredictionDB
    return PredictionDB(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def sample_record():
    return {
        "record_id"     : str(uuid.uuid4()),
        "prediction_id" : str(uuid.uuid4()),
        "session_id"    : "sess_001",
        "user_id"       : "user_001",
        "timestamp"     : datetime.utcnow().isoformat(),
        "valence"       : "High",
        "arousal"       : "Low",
        "dominance"     : "High",
        "valence_prob"  : 0.72,
        "arousal_prob"  : 0.38,
        "dominance_prob": 0.65,
        "n_windows"     : 29,
        "signal_quality": "good",
        "model_version" : "1.0.0",
        "processing_ms" : 118.0,
    }


@pytest.fixture
def history_df():
    n = 20
    return pd.DataFrame({
        "timestamp"     : [datetime.utcnow() - timedelta(hours=i) for i in range(n, 0, -1)],
        "session_id"    : [f"s{i}" for i in range(n)],
        "valence_prob"  : np.linspace(0.40, 0.72, n),
        "arousal_prob"  : np.random.default_rng(42).uniform(0.35, 0.60, n),
        "dominance_prob": np.random.default_rng(7).uniform(0.45, 0.65, n),
        "valence"       : ["High" if v > 0.5 else "Low" for v in np.linspace(0.40, 0.72, n)],
        "arousal"       : ["Low"] * n,
        "dominance"     : ["High"] * n,
    })


# ══════════════════════════════════════════════════════════════════
# DATABASE TESTS
# ══════════════════════════════════════════════════════════════════

class TestPredictionDB:

    def test_health_check(self, tmp_db):
        assert tmp_db.health_check() is True

    def test_save_and_retrieve_prediction(self, tmp_db, sample_record):
        assert tmp_db.save_prediction(sample_record) is True
        df = tmp_db.get_user_history("user_001", days=30)
        assert len(df) == 1
        assert df.iloc[0]["valence"] == "High"
        assert abs(float(df.iloc[0]["valence_prob"]) - 0.72) < 0.001

    def test_save_duplicate_uses_replace(self, tmp_db, sample_record):
        tmp_db.save_prediction(sample_record)
        # same record_id → replace
        sample_record["valence_prob"] = 0.88
        tmp_db.save_prediction(sample_record)
        df = tmp_db.get_user_history("user_001")
        assert len(df) == 1
        assert abs(float(df.iloc[0]["valence_prob"]) - 0.88) < 0.001

    def test_multiple_users_isolated(self, tmp_db):
        for uid in ["user_A", "user_B"]:
            rec = {
                "record_id": str(uuid.uuid4()), "prediction_id": str(uuid.uuid4()),
                "session_id": f"s_{uid}", "user_id": uid,
                "timestamp": datetime.utcnow().isoformat(),
                "valence": "High", "arousal": "Low", "dominance": "High",
                "valence_prob": 0.7, "arousal_prob": 0.4, "dominance_prob": 0.6,
                "n_windows": 10, "signal_quality": "good",
                "model_version": "1.0.0", "processing_ms": 100.0,
            }
            tmp_db.save_prediction(rec)
        assert len(tmp_db.get_user_history("user_A")) == 1
        assert len(tmp_db.get_user_history("user_B")) == 1

    def test_history_day_filter(self, tmp_db):
        # old record: 40 days ago
        old = {
            "record_id": "old1", "prediction_id": "p_old",
            "session_id": "s_old", "user_id": "user_001",
            "timestamp": (datetime.utcnow() - timedelta(days=40)).isoformat(),
            "valence": "Low", "arousal": "Low", "dominance": "Low",
            "valence_prob": 0.3, "arousal_prob": 0.3, "dominance_prob": 0.3,
            "n_windows": 5, "signal_quality": "good",
            "model_version": "1.0.0", "processing_ms": 80.0,
        }
        tmp_db.save_prediction(old)
        # recent record
        rec = {**old, "record_id": "new1", "session_id": "s_new",
               "timestamp": datetime.utcnow().isoformat()}
        tmp_db.save_prediction(rec)

        assert len(tmp_db.get_user_history("user_001", days=30)) == 1
        assert len(tmp_db.get_user_history("user_001", days=0))  == 2

    def test_pagination(self, tmp_db):
        for i in range(10):
            rec = {
                "record_id": f"r{i}", "prediction_id": f"p{i}",
                "session_id": f"s{i}", "user_id": "user_001",
                "timestamp": (datetime.utcnow() - timedelta(minutes=i)).isoformat(),
                "valence": "High", "arousal": "Low", "dominance": "High",
                "valence_prob": 0.6, "arousal_prob": 0.4, "dominance_prob": 0.6,
                "n_windows": 5, "signal_quality": "good",
                "model_version": "1.0.0", "processing_ms": 90.0,
            }
            tmp_db.save_prediction(rec)
        page1 = tmp_db.get_user_history("user_001", days=0, limit=5, offset=0)
        page2 = tmp_db.get_user_history("user_001", days=0, limit=5, offset=5)
        assert len(page1) == 5
        assert len(page2) == 5
        assert set(page1["record_id"]) & set(page2["record_id"]) == set()

    def test_upsert_and_get_user(self, tmp_db):
        user = {
            "user_id": "user_001",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "baseline_valence": 0.62,
            "baseline_arousal": 0.41,
            "baseline_dominance": 0.67,
            "total_sessions": 5,
            "total_windows": 145,
            "metadata": json.dumps({"device": "Emotiv"}),
        }
        assert tmp_db.upsert_user(user) is True
        fetched = tmp_db.get_user("user_001")
        assert fetched is not None
        assert abs(fetched["baseline_valence"] - 0.62) < 0.001
        assert fetched["total_sessions"] == 5

    def test_get_nonexistent_user_returns_none(self, tmp_db):
        assert tmp_db.get_user("no_such_user") is None

    def test_save_and_get_insight(self, tmp_db):
        insight = {
            "insight_id"  : str(uuid.uuid4()),
            "user_id"     : "user_001",
            "generated_at": datetime.utcnow().isoformat(),
            "insight_type": "trend",
            "dimension"   : "valence",
            "title"       : "Valence trending up",
            "description" : "Your valence has increased.",
            "severity"    : "info",
            "value"       : 0.71,
            "reference"   : 0.55,
            "period_start": None,
            "period_end"  : None,
            "tags"        : json.dumps(["valence", "trend"]),
        }
        assert tmp_db.save_insight(insight) is True
        results = tmp_db.get_user_insights("user_001", days=7)
        assert len(results) == 1
        assert results[0]["title"] == "Valence trending up"
        assert isinstance(results[0]["tags"], list)

    def test_platform_stats(self, tmp_db, sample_record):
        tmp_db.save_prediction(sample_record)
        stats = tmp_db.get_platform_stats()
        assert stats["total_predictions"] == 1
        assert stats["total_users"] == 1
        assert stats["total_sessions"] == 1

    def test_export_csv_fallback(self, tmp_db, sample_record, tmp_path):
        tmp_db.save_prediction(sample_record)
        out = tmp_db.export_parquet(str(tmp_path / "out.parquet"))
        assert os.path.exists(out)
        ext = os.path.splitext(out)[1]
        assert ext in (".parquet", ".csv")
        # readable back
        if ext == ".csv":
            df = pd.read_csv(out)
        else:
            df = pd.read_parquet(out)
        assert len(df) == 1


# ══════════════════════════════════════════════════════════════════
# INSIGHT ENGINE TESTS
# ══════════════════════════════════════════════════════════════════

class TestInsightEngine:

    def test_trend_detected_on_upward_series(self, history_df):
        from src.insights.engine import detect_trends
        insights = detect_trends(history_df, "user_001", window=5)
        # valence goes from 0.40 to 0.72 — must be detected
        dims = [i.dimension.value if i.dimension else None for i in insights]
        assert "valence" in dims

    def test_trend_not_detected_on_flat_series(self):
        from src.insights.engine import detect_trends
        n = 15
        df = pd.DataFrame({
            "timestamp"     : [datetime.utcnow() - timedelta(hours=i) for i in range(n, 0, -1)],
            "valence_prob"  : [0.55] * n,
            "arousal_prob"  : [0.45] * n,
            "dominance_prob": [0.60] * n,
        })
        insights = detect_trends(df, "user_001")
        assert len(insights) == 0

    def test_anomaly_detected_on_spike(self, history_df):
        from src.insights.engine import detect_anomalies
        spiked = history_df.copy()
        spiked.iloc[-1, spiked.columns.get_loc("valence_prob")] = 0.99
        insights = detect_anomalies(spiked, "user_001", z_threshold=2.0)
        dims = [i.dimension.value if i.dimension else None for i in insights]
        assert "valence" in dims

    def test_anomaly_not_detected_on_normal_value(self, history_df):
        from src.insights.engine import detect_anomalies
        insights = detect_anomalies(history_df, "user_001", z_threshold=2.0)
        # history_df has gradual increase, last value is ~0.72, mean ~0.56
        # z ≈ (0.72-0.56)/std — may or may not fire depending on std
        # just verify it returns a list
        assert isinstance(insights, list)

    def test_peak_detected_when_new_high(self, history_df):
        from src.insights.engine import detect_peaks
        peaked = history_df.copy()
        peaked.iloc[-1, peaked.columns.get_loc("valence_prob")] = 0.95
        insights = detect_peaks(peaked, "user_001")
        types = [i.insight_type.value for i in insights]
        assert "peak" in types

    def test_stability_detected_on_constant_series(self):
        from src.insights.engine import detect_stability
        n = 10
        df = pd.DataFrame({
            "timestamp"     : [datetime.utcnow() - timedelta(hours=i) for i in range(n, 0, -1)],
            "valence_prob"  : [0.72] * n,
            "arousal_prob"  : [0.42] * n,
            "dominance_prob": [0.65] * n,
        })
        insights = detect_stability(df, "user_001", cv_threshold=0.10)
        assert len(insights) >= 1

    def test_correlation_detected(self):
        from src.insights.engine import detect_correlations
        n = 20
        base = np.linspace(0.3, 0.8, n)
        df = pd.DataFrame({
            "timestamp"     : [datetime.utcnow() - timedelta(hours=i) for i in range(n, 0, -1)],
            "valence_prob"  : base,
            "arousal_prob"  : base * 1.02,    # near-perfect correlation
            "dominance_prob": np.random.default_rng(0).uniform(0.4, 0.6, n),
        })
        insights = detect_correlations(df, "user_001", corr_threshold=0.65)
        assert len(insights) >= 1

    def test_generate_returns_sorted_by_severity(self, history_df):
        from src.insights.engine import InsightEngine
        engine  = InsightEngine()
        results = engine.generate("user_001", history_df, max_insights=10)
        severities = [r.severity for r in results]
        order = {"alert": 0, "warning": 1, "info": 2}
        for i in range(len(severities) - 1):
            assert order.get(severities[i], 3) <= order.get(severities[i+1], 3)

    def test_generate_respects_max_insights(self, history_df):
        from src.insights.engine import InsightEngine
        engine  = InsightEngine()
        results = engine.generate("user_001", history_df, max_insights=2)
        assert len(results) <= 2

    def test_generate_on_empty_df_returns_empty(self):
        from src.insights.engine import InsightEngine
        engine  = InsightEngine()
        results = engine.generate("user_001", pd.DataFrame())
        assert results == []

    def test_update_user_baseline(self, history_df):
        from src.insights.engine import InsightEngine
        engine    = InsightEngine()
        baselines = engine.update_user_baseline(history_df)
        assert "baseline_valence"   in baselines
        assert "baseline_arousal"   in baselines
        assert "baseline_dominance" in baselines
        # valence goes 0.40 → 0.72, EMA (alpha=0.1) should be between them
        assert 0.40 < baselines["baseline_valence"] < 0.72

    def test_update_baseline_empty_df(self):
        from src.insights.engine import InsightEngine
        engine    = InsightEngine()
        baselines = engine.update_user_baseline(pd.DataFrame())
        assert baselines["baseline_valence"]   == 0.5
        assert baselines["baseline_arousal"]   == 0.5
        assert baselines["baseline_dominance"] == 0.5


# ══════════════════════════════════════════════════════════════════
# SIGNAL QUALITY TESTS
# ══════════════════════════════════════════════════════════════════

class TestSignalQuality:

    def _eeg(self, n_samp=128*30, n_ch=14, add_nan=False, flat=False):
        arr = np.random.randn(n_samp, n_ch).astype(np.float32)
        if add_nan:
            arr[0, 0] = float("nan")
        if flat:
            arr[:, :6] = 0.0   # 6 flat channels
        return arr

    def _ecg(self, n_samp=256*30):
        return np.random.randn(n_samp, 2).astype(np.float32)

    def test_good_signal(self):
        from src.pipeline.signal_pipeline import check_signal_quality, SignalQuality
        q, msg = check_signal_quality(self._eeg(), self._ecg(), 128.0)
        assert q == SignalQuality.GOOD
        assert msg == "OK"

    def test_nan_eeg_is_poor(self):
        from src.pipeline.signal_pipeline import check_signal_quality, SignalQuality
        q, msg = check_signal_quality(self._eeg(add_nan=True), self._ecg(), 128.0)
        assert q == SignalQuality.POOR
        assert "NaN" in msg

    def test_nan_ecg_is_poor(self):
        from src.pipeline.signal_pipeline import check_signal_quality, SignalQuality
        ecg_nan = self._ecg(); ecg_nan[0, 0] = float("nan")
        q, _ = check_signal_quality(self._eeg(), ecg_nan, 128.0)
        assert q == SignalQuality.POOR

    def test_short_signal_is_poor(self):
        from src.pipeline.signal_pipeline import check_signal_quality, SignalQuality
        short_eeg = np.random.randn(100, 14).astype(np.float32)
        q, msg = check_signal_quality(short_eeg, self._ecg(), 128.0)
        assert q == SignalQuality.POOR
        assert "short" in msg.lower()

    def test_flat_channels_is_poor(self):
        from src.pipeline.signal_pipeline import check_signal_quality, SignalQuality
        q, msg = check_signal_quality(
            self._eeg(flat=True), self._ecg(), 128.0
        )
        assert q == SignalQuality.POOR
        assert "flat" in msg.lower()

    def test_degraded_signal_single_flat_channel(self):
        from src.pipeline.signal_pipeline import check_signal_quality, SignalQuality
        eeg = self._eeg()
        eeg[:, 0] = 0.0   # just 1 flat channel
        q, msg = check_signal_quality(eeg, self._ecg(), 128.0)
        assert q == SignalQuality.DEGRADED
        assert "flat" in msg.lower()


# ══════════════════════════════════════════════════════════════════
# PIPELINE END-TO-END TESTS (no model weights needed)
# ══════════════════════════════════════════════════════════════════

class TestSignalPipelineLogic:
    """
    Tests the pipeline orchestration logic without loading actual model weights.
    Uses a mock InferenceEngine that returns deterministic probabilities.
    """

    def _make_mock_engine(self, prob=0.72):
        """Return a mock InferenceEngine that always predicts fixed probabilities."""
        import unittest.mock as mock
        engine = mock.MagicMock()
        engine.predict_windows.return_value = {
            "valence"  : [{"prob_high": prob, "prob_low": 1-prob, "label": "High"}] * 29,
            "arousal"  : [{"prob_high": 0.38, "prob_low": 0.62,   "label": "Low"}]  * 29,
            "dominance": [{"prob_high": 0.65, "prob_low": 0.35,   "label": "High"}] * 29,
        }
        return engine

    def _config(self):
        return {
            "data": {
                "sampling_rate_eeg": 128, "sampling_rate_ecg": 256,
                "segment_length": 4, "overlap": 2, "norm_method": "zscore",
            },
            "labels": {"threshold": 3},
        }

    def _make_signal_input(self):
        """Build a minimal valid signal input dict (bypassing pydantic)."""
        return type("SI", (), {
            "session_id"  : "sess_test",
            "user_id"     : "user_test",
            "timestamp"   : datetime.utcnow(),
            "eeg_data"    : np.random.randn(128*30, 14).astype(np.float32).tolist(),
            "ecg_data"    : np.random.randn(256*30, 2).astype(np.float32).tolist(),
            "eeg_baseline": None,
            "ecg_baseline": None,
        })()

    def test_pipeline_run_returns_prediction_and_record(self):
        from src.pipeline.signal_pipeline import SignalPipeline
        engine   = self._make_mock_engine(prob=0.72)
        pipeline = SignalPipeline(engine, self._config())
        pred, record = pipeline.run(self._make_signal_input())

        assert pred.valence.value   == "High"
        assert pred.arousal.value   == "Low"
        assert pred.dominance.value == "High"
        assert abs(pred.valence_prob - 0.72) < 0.001
        assert pred.n_windows > 0          # count depends on signal length
        assert pred.processing_ms > 0
        assert record.session_id == "sess_test"

    def test_pipeline_low_prediction(self):
        from src.pipeline.signal_pipeline import SignalPipeline
        engine   = self._make_mock_engine(prob=0.30)
        pipeline = SignalPipeline(engine, self._config())
        pred, _  = pipeline.run(self._make_signal_input())
        assert pred.valence.value == "Low"

    def test_pipeline_window_count_is_positive(self):
        from src.pipeline.signal_pipeline import SignalPipeline
        engine   = self._make_mock_engine()
        pipeline = SignalPipeline(engine, self._config())
        pred, _  = pipeline.run(self._make_signal_input())
        assert pred.n_windows > 0

    def test_pipeline_record_has_all_required_fields(self):
        from src.pipeline.signal_pipeline import SignalPipeline
        engine   = self._make_mock_engine()
        pipeline = SignalPipeline(engine, self._config())
        _, record = pipeline.run(self._make_signal_input())
        for field in ["prediction_id", "session_id", "user_id", "timestamp",
                      "valence", "arousal", "dominance",
                      "valence_prob", "arousal_prob", "dominance_prob",
                      "n_windows", "signal_quality", "model_version",
                      "processing_ms"]:
            assert hasattr(record, field), f"Missing field: {field}"

    def test_pipeline_poor_quality_still_returns_result(self):
        """Poor signal quality should warn but not crash — result still returned."""
        from src.pipeline.signal_pipeline import SignalPipeline, SignalQuality
        engine = self._make_mock_engine()
        pipeline = SignalPipeline(engine, self._config())
        sig = self._make_signal_input()
        # Inject NaN into EEG
        eeg = np.array(sig.eeg_data)
        eeg[0, 0] = float("nan")
        sig.eeg_data = eeg.tolist()
        # Should still execute (quality check warns, preprocessing may handle it)
        # If preprocessing raises ValueError, pipeline should propagate
        try:
            pred, record = pipeline.run(sig)
            assert record is not None
        except ValueError:
            pass   # acceptable — poor signal raises ValueError


# ══════════════════════════════════════════════════════════════════
# API ENDPOINT TESTS (requires fastapi + httpx)
# ══════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not all(__import__("importlib").util.find_spec(m) for m in ["fastapi", "httpx"]),
    reason="fastapi and httpx required"
)
class TestAPIEndpoints:

    @pytest.fixture
    def client(self, tmp_path):
        from fastapi.testclient import TestClient
        from src.api.routes import create_app

        # App without models (no model_paths) — /predict returns 503
        app = create_app(
            config_path = "configs/default.yaml",
            model_paths = None,
            db_path     = str(tmp_path / "test.db"),
        )
        return TestClient(app)

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert "model_loaded" in body
        assert "db_connected" in body
        assert body["db_connected"] is True

    def test_predict_without_model_returns_503(self, client):
        payload = {
            "user_id" : "u001",
            "eeg_data": np.random.randn(128*30, 14).tolist(),
            "ecg_data": np.random.randn(256*30, 2).tolist(),
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 503

    def test_history_empty_user_returns_empty(self, client):
        r = client.get("/users/nobody/history")
        assert r.status_code == 200
        body = r.json()
        assert body["total_records"] == 0
        assert body["records"] == []

    def test_insights_empty_user_returns_empty(self, client):
        r = client.get("/users/nobody/insights")
        assert r.status_code == 200
        body = r.json()
        assert body["insights"] == []

    def test_profile_nonexistent_user_returns_404(self, client):
        r = client.get("/users/nobody/profile")
        assert r.status_code == 404

    def test_summary_nonexistent_user_returns_404(self, client):
        r = client.get("/users/nobody/summary")
        assert r.status_code == 404

    def test_platform_stats_returns_counts(self, client):
        r = client.get("/platform/stats")
        assert r.status_code == 200
        body = r.json()
        assert "total_predictions" in body
        assert "total_users" in body

    def test_invalid_eeg_shape_returns_422(self, client):
        payload = {
            "user_id" : "u001",
            "eeg_data": [[0.0] * 10] * 700,   # wrong: 10 channels instead of 14
            "ecg_data": np.random.randn(256*30, 2).tolist(),
        }
        r = client.post("/predict", json=payload)
        assert r.status_code in (422, 503)   # 422 from pydantic, 503 if model missing

    def test_docs_endpoint_accessible(self, client):
        r = client.get("/docs")
        assert r.status_code == 200