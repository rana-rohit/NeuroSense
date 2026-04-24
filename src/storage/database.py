"""
storage/database.py
Persistent prediction store.

Primary:  SQLite (zero-dependency, production-upgradeable to Postgres)
Export:   Parquet via pandas for analytics / insight engine
Schema:   Matches PredictionRecord in schemas/models.py

Upgrade path: swap SQLite → Postgres by changing DATABASE_URL env var.
"""

import os
import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from contextlib import contextmanager

from src.utils.logger import get_logger

logger = get_logger("database")

DEFAULT_DB_PATH = "outputs/platform/predictions.db"


# ── Schema DDL ────────────────────────────────────────────────────

_CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    record_id       TEXT PRIMARY KEY,
    prediction_id   TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    timestamp       TEXT NOT NULL,

    valence         TEXT NOT NULL,
    arousal         TEXT NOT NULL,
    dominance       TEXT NOT NULL,

    valence_prob    REAL NOT NULL,
    arousal_prob    REAL NOT NULL,
    dominance_prob  REAL NOT NULL,

    n_windows       INTEGER NOT NULL,
    signal_quality  TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    processing_ms   REAL NOT NULL,

    created_at      TEXT DEFAULT (datetime('now'))
);
"""

_CREATE_USERS = """
CREATE TABLE IF NOT EXISTS users (
    user_id             TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    baseline_valence    REAL DEFAULT 0.5,
    baseline_arousal    REAL DEFAULT 0.5,
    baseline_dominance  REAL DEFAULT 0.5,
    total_sessions      INTEGER DEFAULT 0,
    total_windows       INTEGER DEFAULT 0,
    metadata            TEXT DEFAULT '{}'
);
"""

_CREATE_INSIGHTS = """
CREATE TABLE IF NOT EXISTS insights (
    insight_id      TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    generated_at    TEXT NOT NULL,
    insight_type    TEXT NOT NULL,
    dimension       TEXT,
    title           TEXT NOT NULL,
    description     TEXT NOT NULL,
    severity        TEXT DEFAULT 'info',
    value           REAL,
    reference       REAL,
    period_start    TEXT,
    period_end      TEXT,
    tags            TEXT DEFAULT '[]'
);
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pred_user ON predictions(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_pred_ts   ON predictions(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_pred_sess ON predictions(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_ins_user  ON insights(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_ins_ts    ON insights(generated_at);",
]


# ── Database class ────────────────────────────────────────────────

class PredictionDB:
    """
    SQLite-backed prediction store.

    Usage:
        db = PredictionDB()
        db.save_prediction(record)
        df = db.get_user_history("user_001", days=7)
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_schema()
        logger.info(f"PredictionDB initialised → {db_path}")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as conn:
            conn.execute(_CREATE_PREDICTIONS)
            conn.execute(_CREATE_USERS)
            conn.execute(_CREATE_INSIGHTS)
            for idx in _INDEXES:
                conn.execute(idx)
        logger.info("Schema initialised")

    # ── Predictions ───────────────────────────────────────────────

    def save_prediction(self, record: dict) -> bool:
        """
        Insert one PredictionRecord dict.
        Returns True on success.
        """
        sql = """
        INSERT OR REPLACE INTO predictions
            (record_id, prediction_id, session_id, user_id, timestamp,
             valence, arousal, dominance,
             valence_prob, arousal_prob, dominance_prob,
             n_windows, signal_quality, model_version, processing_ms)
        VALUES
            (:record_id, :prediction_id, :session_id, :user_id, :timestamp,
             :valence, :arousal, :dominance,
             :valence_prob, :arousal_prob, :dominance_prob,
             :n_windows, :signal_quality, :model_version, :processing_ms)
        """
        try:
            # normalise timestamp
            if isinstance(record.get("timestamp"), datetime):
                record = dict(record)
                record["timestamp"] = record["timestamp"].isoformat()
            with self._conn() as conn:
                conn.execute(sql, record)
            logger.debug(f"Saved prediction {record['prediction_id']}")
            return True
        except Exception as e:
            logger.error(f"save_prediction failed: {e}")
            return False

    def get_user_history(
        self,
        user_id  : str,
        days     : int = 30,
        limit    : int = 1000,
        offset   : int = 0,
    ) -> pd.DataFrame:
        """
        Return prediction history for a user as a DataFrame.

        Args:
            user_id: user identifier
            days   : look-back window (0 = all time)
            limit  : max rows
            offset : pagination offset
        """
        since = (
            (datetime.utcnow() - timedelta(days=days)).isoformat()
            if days > 0 else "1970-01-01"
        )
        sql = """
        SELECT * FROM predictions
        WHERE user_id = ? AND timestamp >= ?
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
        """
        with self._conn() as conn:
            rows = conn.execute(sql, (user_id, since, limit, offset)).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([dict(r) for r in rows])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def get_session(self, session_id: str) -> Optional[dict]:
        """Return all records for a specific session."""
        sql = "SELECT * FROM predictions WHERE session_id = ? ORDER BY timestamp"
        with self._conn() as conn:
            rows = conn.execute(sql, (session_id,)).fetchall()
        return [dict(r) for r in rows] if rows else None

    def count_user_sessions(self, user_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT session_id) as n FROM predictions WHERE user_id=?",
                (user_id,)
            ).fetchone()
        return row["n"] if row else 0

    # ── Users ─────────────────────────────────────────────────────

    def upsert_user(self, user: dict) -> bool:
        sql = """
        INSERT INTO users
            (user_id, created_at, updated_at,
             baseline_valence, baseline_arousal, baseline_dominance,
             total_sessions, total_windows, metadata)
        VALUES
            (:user_id, :created_at, :updated_at,
             :baseline_valence, :baseline_arousal, :baseline_dominance,
             :total_sessions, :total_windows, :metadata)
        ON CONFLICT(user_id) DO UPDATE SET
            updated_at          = excluded.updated_at,
            baseline_valence    = excluded.baseline_valence,
            baseline_arousal    = excluded.baseline_arousal,
            baseline_dominance  = excluded.baseline_dominance,
            total_sessions      = excluded.total_sessions,
            total_windows       = excluded.total_windows,
            metadata            = excluded.metadata
        """
        try:
            u = dict(user)
            for k in ("created_at", "updated_at"):
                if isinstance(u.get(k), datetime):
                    u[k] = u[k].isoformat()
            if isinstance(u.get("metadata"), dict):
                u["metadata"] = json.dumps(u["metadata"])
            with self._conn() as conn:
                conn.execute(sql, u)
            return True
        except Exception as e:
            logger.error(f"upsert_user failed: {e}")
            return False

    def get_user(self, user_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id=?", (user_id,)
            ).fetchone()
        if not row:
            return None
        u = dict(row)
        u["metadata"] = json.loads(u.get("metadata", "{}"))
        return u

    # ── Insights ──────────────────────────────────────────────────

    def save_insight(self, insight: dict) -> bool:
        sql = """
        INSERT OR REPLACE INTO insights
            (insight_id, user_id, generated_at, insight_type,
             dimension, title, description, severity,
             value, reference, period_start, period_end, tags)
        VALUES
            (:insight_id, :user_id, :generated_at, :insight_type,
             :dimension, :title, :description, :severity,
             :value, :reference, :period_start, :period_end, :tags)
        """
        try:
            ins = dict(insight)
            for k in ("generated_at", "period_start", "period_end"):
                if isinstance(ins.get(k), datetime):
                    ins[k] = ins[k].isoformat()
            if isinstance(ins.get("tags"), list):
                ins["tags"] = json.dumps(ins["tags"])
            with self._conn() as conn:
                conn.execute(sql, ins)
            return True
        except Exception as e:
            logger.error(f"save_insight failed: {e}")
            return False

    def get_user_insights(self, user_id: str,
                           days: int = 7) -> List[dict]:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        sql = """
        SELECT * FROM insights
        WHERE user_id=? AND generated_at >= ?
        ORDER BY generated_at DESC
        """
        with self._conn() as conn:
            rows = conn.execute(sql, (user_id, since)).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["tags"] = json.loads(d.get("tags", "[]"))
            results.append(d)
        return results

    # ── Analytics exports ─────────────────────────────────────────

    def export_parquet(self, out_path: str = "outputs/platform/predictions.parquet"):
        """
        Export full predictions table to Parquet (preferred) or CSV fallback.
        Parquet requires pyarrow or fastparquet. CSV is always available.
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with self._conn() as conn:
            df = pd.read_sql("SELECT * FROM predictions", conn)
        try:
            df.to_parquet(out_path, index=False)
            logger.info(f"Exported {len(df)} rows → {out_path} (parquet)")
        except ImportError:
            csv_path = out_path.replace(".parquet", ".csv")
            df.to_csv(csv_path, index=False)
            out_path = csv_path
            logger.warning(
                f"pyarrow not installed — exported as CSV → {csv_path}. "
                "Install pyarrow for Parquet: pip install pyarrow"
            )
        return out_path

    def get_platform_stats(self) -> Dict:
        """High-level platform statistics."""
        with self._conn() as conn:
            stats = {
                "total_predictions": conn.execute(
                    "SELECT COUNT(*) FROM predictions").fetchone()[0],
                "total_users": conn.execute(
                    "SELECT COUNT(DISTINCT user_id) FROM predictions").fetchone()[0],
                "total_sessions": conn.execute(
                    "SELECT COUNT(DISTINCT session_id) FROM predictions").fetchone()[0],
                "total_insights": conn.execute(
                    "SELECT COUNT(*) FROM insights").fetchone()[0],
            }
        return stats

    def health_check(self) -> bool:
        try:
            with self._conn() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False