"""
api/routes.py
FastAPI application — Emotion Intelligence Platform API.

Endpoints:
  GET  /health                          — liveness + readiness
  POST /predict                         — submit EEG/ECG → emotion prediction
  GET  /users/{user_id}/history         — prediction history with pagination
  GET  /users/{user_id}/insights        — generated insights
  GET  /users/{user_id}/profile         — user emotional baseline profile
  GET  /users/{user_id}/summary         — aggregated stats
  GET  /platform/stats                  — admin platform-wide statistics
  POST /admin/export                    — export DB to Parquet

Run locally:
    uvicorn src.api.routes:app --reload --port 8000

Run on Colab (with ngrok):
    !pip install fastapi uvicorn pyngrok pydantic -q
    from pyngrok import ngrok
    ngrok.set_auth_token("YOUR_TOKEN")
    public_url = ngrok.connect(8000)
    !uvicorn src.api.routes:app --host 0.0.0.0 --port 8000 &
"""

import os
import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.schemas.models  import (
    SignalInput, PredictResponse, HistoryResponse,
    InsightResponse, HealthResponse, PredictionRecord,
)
from src.storage.database   import PredictionDB
from src.insights.engine    import InsightEngine
from src.utils.logger       import get_logger
from src.utils.config       import load_config

logger   = get_logger("api")
_START   = time.time()

# ── Application factory ───────────────────────────────────────────

def create_app(
    config_path  : str = "configs/default.yaml",
    model_paths  : Optional[dict] = None,
    model_type   : str = "fusion",
    db_path      : str = "outputs/platform/predictions.db",
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        config_path: YAML config file path
        model_paths: {"valence": "path.pt", ...} — None = no model loaded
        model_type : deep or baseline model type
        db_path    : SQLite database path
    """
    cfg = load_config(config_path)

    app = FastAPI(
        title       = "Emotion Intelligence Platform",
        description = "EEG + ECG physiological signal analysis API",
        version     = "1.0.0",
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins  = ["*"],
        allow_methods  = ["*"],
        allow_headers  = ["*"],
    )

    # Shared state
    app.state.config        = cfg
    app.state.db            = PredictionDB(db_path)
    app.state.insight_engine= InsightEngine()
    app.state.pipeline      = None
    app.state.model_loaded  = False

    # Load inference engine if paths provided
    if model_paths:
        try:
            from src.pipeline.signal_pipeline import SignalPipeline, InferenceEngine
            engine = InferenceEngine(model_paths, model_type, cfg)
            app.state.pipeline    = SignalPipeline(engine, cfg)
            app.state.model_loaded= True
            logger.info("Inference engine loaded ✅")
        except Exception as e:
            logger.warning(f"Could not load model: {e}. /predict will be unavailable.")

    # ── Dependencies ──────────────────────────────────────────────

    def get_db() -> PredictionDB:
        return app.state.db

    def get_pipeline():
        if not app.state.pipeline:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Start server with model_paths configured."
            )
        return app.state.pipeline

    def get_insight_engine() -> InsightEngine:
        return app.state.insight_engine

    # ══════════════════════════════════════════════════════════════
    # ROUTES
    # ══════════════════════════════════════════════════════════════

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health():
        """
        Liveness + readiness check.

        Returns platform health including model and DB status.
        """
        return HealthResponse(
            status         = "healthy",
            version        = "1.0.0",
            model_loaded   = app.state.model_loaded,
            db_connected   = app.state.db.health_check(),
            uptime_seconds = round(time.time() - _START, 1),
        )

    # ── Predict ───────────────────────────────────────────────────

    @app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
    async def predict(
        signal  : SignalInput,
        pipeline = Depends(get_pipeline),
        db       = Depends(get_db),
        ie       = Depends(get_insight_engine),
    ):
        """
        Submit EEG + ECG signals for emotion prediction.

        Flow:
          validate → preprocess → infer → store → generate insights → respond

        Body: SignalInput JSON
        Returns: EmotionPrediction + realtime insights
        """
        try:
            pred, record = pipeline.run(signal)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise HTTPException(status_code=500, detail="Internal pipeline error")

        # Persist prediction
        db.save_prediction(record.model_dump())

        # Update user profile
        user = db.get_user(signal.user_id) or {
            "user_id"           : signal.user_id,
            "created_at"        : datetime.utcnow(),
            "updated_at"        : datetime.utcnow(),
            "baseline_valence"  : 0.5,
            "baseline_arousal"  : 0.5,
            "baseline_dominance": 0.5,
            "total_sessions"    : 0,
            "total_windows"     : 0,
            "metadata"          : {},
        }
        user["total_sessions"] += 1
        user["total_windows"]  += pred.n_windows
        user["updated_at"]      = datetime.utcnow()
        db.upsert_user(user)

        # Generate realtime insights from history
        history_df = db.get_user_history(signal.user_id, days=30)
        insights   = ie.generate(signal.user_id, history_df, max_insights=5)

        # Persist new insights
        for ins in insights:
            db.save_insight(ins.model_dump())

        return PredictResponse(
            prediction = pred,
            insights   = insights,
            message    = (
                f"Processed {pred.n_windows} windows in {pred.processing_ms}ms. "
                f"Signal quality: {pred.signal_quality.value}."
            ),
        )

    # ── History ───────────────────────────────────────────────────

    @app.get(
        "/users/{user_id}/history",
        response_model=HistoryResponse,
        tags=["Users"],
    )
    async def get_history(
        user_id  : str,
        days     : int  = Query(default=30, ge=1, le=365),
        page     : int  = Query(default=1, ge=1),
        page_size: int  = Query(default=50, ge=1, le=200),
        db       = Depends(get_db),
    ):
        """
        Retrieve paginated prediction history for a user.

        Args:
            days     : look-back window in days
            page     : page number (1-indexed)
            page_size: records per page
        """
        offset = (page - 1) * page_size
        df     = db.get_user_history(user_id, days=days,
                                      limit=page_size, offset=offset)
        if df.empty:
            return HistoryResponse(
                user_id=user_id, total_records=0, records=[],
                page=page, page_size=page_size)

        records = [
            PredictionRecord(**row)
            for row in df.to_dict(orient="records")
        ]
        return HistoryResponse(
            user_id      = user_id,
            total_records= len(records),
            records      = records,
            page         = page,
            page_size    = page_size,
        )

    # ── Insights ──────────────────────────────────────────────────

    @app.get(
        "/users/{user_id}/insights",
        response_model=InsightResponse,
        tags=["Users"],
    )
    async def get_insights(
        user_id    : str,
        days       : int = Query(default=7,  ge=1, le=90),
        regenerate : bool = Query(default=False),
        db         = Depends(get_db),
        ie         = Depends(get_insight_engine),
    ):
        """
        Return insights for a user.

        If regenerate=true, recomputes from full history.
        Otherwise returns cached insights from the last `days`.
        """
        if regenerate:
            history_df = db.get_user_history(user_id, days=90)
            insights_raw = ie.generate(user_id, history_df)
            for ins in insights_raw:
                db.save_insight(ins.model_dump())
        else:
            raw  = db.get_user_insights(user_id, days=days)
            from src.schemas.models import Insight
            insights_raw = [Insight(**r) for r in raw]

        return InsightResponse(
            user_id     = user_id,
            period_days = days,
            insights    = insights_raw,
        )

    # ── User profile ──────────────────────────────────────────────

    @app.get("/users/{user_id}/profile", tags=["Users"])
    async def get_profile(user_id: str, db = Depends(get_db)):
        """Return user emotional baseline profile."""
        user = db.get_user(user_id)
        if not user:
            raise HTTPException(
                status_code=404,
                detail=f"User '{user_id}' not found"
            )
        return user

    # ── Summary ───────────────────────────────────────────────────

    @app.get("/users/{user_id}/summary", tags=["Users"])
    async def get_summary(
        user_id : str,
        days    : int = Query(default=30, ge=1, le=365),
        db      = Depends(get_db),
    ):
        """
        Aggregated emotional summary for a user.

        Returns mean and std of each emotion dimension probability
        and session count over the requested period.
        """
        df = db.get_user_history(user_id, days=days)
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data for user '{user_id}' in the last {days} days"
            )

        import pandas as pd
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        summary = {
            "user_id"       : user_id,
            "period_days"   : days,
            "total_sessions": int(df["session_id"].nunique()),
            "total_windows" : int(df["n_windows"].sum()),
        }
        for dim in ["valence", "arousal", "dominance"]:
            col = f"{dim}_prob"
            summary[f"{dim}_mean"] = round(float(df[col].mean()), 4)
            summary[f"{dim}_std"]  = round(float(df[col].std()),  4)
            summary[f"{dim}_high_pct"] = round(
                float((df[dim] == "High").mean() * 100), 1)

        return summary

    # ── Platform stats ────────────────────────────────────────────

    @app.get("/platform/stats", tags=["Admin"])
    async def platform_stats(db = Depends(get_db)):
        """Platform-wide aggregate statistics."""
        return db.get_platform_stats()

    @app.post("/admin/export", tags=["Admin"])
    async def export_db(
        out_path: str = Query(default="outputs/platform/predictions.parquet"),
        db = Depends(get_db),
    ):
        """Export prediction database to Parquet."""
        try:
            path = db.export_parquet(out_path)
            return {"status": "success", "path": path}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ── Default app instance ──────────────────────────────────────────

app = create_app()