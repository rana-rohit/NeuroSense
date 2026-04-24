"""
insights/engine.py
Generates actionable insights from a user's prediction history.

Insight types:
  TREND       — sustained directional shift over N sessions
  ANOMALY     — sudden deviation from personal baseline
  PEAK        — highest/lowest recorded emotional state
  STABILITY   — consistent emotional state across sessions
  CORRELATION — co-occurrence of dimensions (e.g. high arousal + low valence)
"""

import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional

from src.utils.logger   import get_logger
from src.schemas.models import Insight, InsightType, EmotionDimension

logger = get_logger("insights")

DIMENSIONS   = ["valence", "arousal", "dominance"]
PROB_COLS    = ["valence_prob", "arousal_prob", "dominance_prob"]


def _make_insight(
    user_id      : str,
    insight_type : InsightType,
    title        : str,
    description  : str,
    severity     : str = "info",
    dimension    : Optional[str] = None,
    value        : Optional[float] = None,
    reference    : Optional[float] = None,
    period_start : Optional[datetime] = None,
    period_end   : Optional[datetime] = None,
    tags         : List[str] = None,
) -> Insight:
    return Insight(
        insight_id   = str(uuid.uuid4()),
        user_id      = user_id,
        generated_at = datetime.utcnow(),
        insight_type = insight_type,
        dimension    = EmotionDimension(dimension) if dimension else None,
        title        = title,
        description  = description,
        severity     = severity,
        value        = round(value, 3) if value is not None else None,
        reference    = round(reference, 3) if reference is not None else None,
        period_start = period_start,
        period_end   = period_end,
        tags         = tags or [],
    )


# ── Individual insight detectors ──────────────────────────────────

def detect_trends(df: pd.DataFrame, user_id: str,
                  window: int = 5) -> List[Insight]:
    """
    Detect sustained directional trends.
    A trend is flagged when the rolling mean shifts > 0.15 over `window` sessions.
    """
    insights = []
    if len(df) < window + 1:
        return insights

    df_sorted = df.sort_values("timestamp").reset_index(drop=True)

    for dim, col in zip(DIMENSIONS, PROB_COLS):
        series = df_sorted[col].values
        if len(series) < window + 1:
            continue

        recent_mean = float(np.mean(series[-window:]))
        prior_mean  = float(np.mean(series[:-window]))
        delta       = recent_mean - prior_mean

        if abs(delta) < 0.12:
            continue

        direction = "increasing" if delta > 0 else "decreasing"
        label     = "High" if delta > 0 else "Low"
        severity  = "warning" if abs(delta) > 0.20 else "info"

        insights.append(_make_insight(
            user_id      = user_id,
            insight_type = InsightType.TREND,
            dimension    = dim,
            title        = f"{dim.capitalize()} trending {direction}",
            description  = (
                f"Your {dim} has been trending {direction} "
                f"over the last {window} sessions. "
                f"Recent average: {recent_mean:.2f} vs "
                f"prior average: {prior_mean:.2f} "
                f"(shift: {delta:+.2f})."
            ),
            severity     = severity,
            value        = recent_mean,
            reference    = prior_mean,
            period_start = df_sorted["timestamp"].iloc[-window],
            period_end   = df_sorted["timestamp"].iloc[-1],
            tags         = [dim, "trend", direction],
        ))

    return insights


def detect_anomalies(df: pd.DataFrame, user_id: str,
                     z_threshold: float = 2.0) -> List[Insight]:
    """
    Flag sessions where a dimension probability deviates > z_threshold
    standard deviations from the user's personal mean.
    """
    insights = []
    if len(df) < 5:
        return insights

    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    latest = df_sorted.iloc[-1]

    for dim, col in zip(DIMENSIONS, PROB_COLS):
        historical = df_sorted[col].values[:-1]
        if len(historical) < 4:
            continue

        mean = float(np.mean(historical))
        std  = float(np.std(historical))
        if std < 0.01:
            continue

        current = float(latest[col])
        z_score = (current - mean) / std

        if abs(z_score) < z_threshold:
            continue

        direction = "spike" if z_score > 0 else "drop"
        severity  = "alert" if abs(z_score) > 3.0 else "warning"

        insights.append(_make_insight(
            user_id      = user_id,
            insight_type = InsightType.ANOMALY,
            dimension    = dim,
            title        = f"Unusual {dim} {direction} detected",
            description  = (
                f"Your latest {dim} reading ({current:.2f}) is "
                f"{abs(z_score):.1f} standard deviations from your "
                f"personal average ({mean:.2f}). "
                f"This is an unusually {'high' if z_score > 0 else 'low'} "
                f"reading for you."
            ),
            severity     = severity,
            value        = current,
            reference    = mean,
            period_start = latest["timestamp"],
            period_end   = latest["timestamp"],
            tags         = [dim, "anomaly", direction],
        ))

    return insights


def detect_peaks(df: pd.DataFrame, user_id: str) -> List[Insight]:
    """
    Detect all-time highs and lows for each dimension.
    Only reported when the new record is set in the latest session.
    """
    insights = []
    if len(df) < 3:
        return insights

    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    latest = df_sorted.iloc[-1]

    for dim, col in zip(DIMENSIONS, PROB_COLS):
        current = float(latest[col])
        historical_max = float(df_sorted[col].iloc[:-1].max())
        historical_min = float(df_sorted[col].iloc[:-1].min())

        if current > historical_max and current > 0.75:
            insights.append(_make_insight(
                user_id      = user_id,
                insight_type = InsightType.PEAK,
                dimension    = dim,
                title        = f"Personal best: highest {dim} recorded",
                description  = (
                    f"Your {dim} score ({current:.2f}) is the highest "
                    f"you have recorded, exceeding your previous best "
                    f"of {historical_max:.2f}."
                ),
                severity     = "info",
                value        = current,
                reference    = historical_max,
                period_start = latest["timestamp"],
                period_end   = latest["timestamp"],
                tags         = [dim, "peak", "high"],
            ))

        if current < historical_min and current < 0.25:
            insights.append(_make_insight(
                user_id      = user_id,
                insight_type = InsightType.PEAK,
                dimension    = dim,
                title        = f"Personal low: lowest {dim} recorded",
                description  = (
                    f"Your {dim} score ({current:.2f}) is the lowest "
                    f"you have recorded, below your previous low "
                    f"of {historical_min:.2f}."
                ),
                severity     = "warning",
                value        = current,
                reference    = historical_min,
                period_start = latest["timestamp"],
                period_end   = latest["timestamp"],
                tags         = [dim, "peak", "low"],
            ))

    return insights


def detect_stability(df: pd.DataFrame, user_id: str,
                     cv_threshold: float = 0.10) -> List[Insight]:
    """
    Flag when a user shows consistently stable emotion over the last 7 days.
    Coefficient of variation (std/mean) < cv_threshold signals stability.
    """
    insights = []
    if len(df) < 5:
        return insights

    week_ago = datetime.utcnow() - timedelta(days=7)
    recent   = df[df["timestamp"] >= week_ago]
    if len(recent) < 4:
        return insights

    for dim, col in zip(DIMENSIONS, PROB_COLS):
        mean = float(recent[col].mean())
        std  = float(recent[col].std())
        cv   = std / (mean + 1e-8)

        if cv > cv_threshold:
            continue

        label = "High" if mean > 0.5 else "Low"
        insights.append(_make_insight(
            user_id      = user_id,
            insight_type = InsightType.STABILITY,
            dimension    = dim,
            title        = f"Stable {label} {dim} this week",
            description  = (
                f"Your {dim} has been remarkably stable over the past week "
                f"(mean={mean:.2f}, std={std:.2f}). "
                f"This indicates a consistently {label.lower()} emotional "
                f"state in this dimension."
            ),
            severity     = "info",
            value        = mean,
            reference    = cv,
            period_start = week_ago,
            period_end   = datetime.utcnow(),
            tags         = [dim, "stability"],
        ))

    return insights


def detect_correlations(df: pd.DataFrame, user_id: str,
                        corr_threshold: float = 0.65) -> List[Insight]:
    """
    Detect strong co-movement between emotion dimensions.
    High valence + high arousal = excited; low valence + high arousal = stressed.
    """
    insights = []
    if len(df) < 8:
        return insights

    pairs = [
        ("valence", "arousal",   "Excited pattern detected",
         "Your high valence and high arousal tend to occur together, "
         "suggesting periods of positive excitement."),
        ("valence", "dominance", "Confident pattern detected",
         "Your positive valence and high dominance co-occur, "
         "suggesting periods of confident, in-control emotional states."),
        ("arousal", "dominance", "Activated dominance pattern detected",
         "Your arousal and sense of control tend to rise together, "
         "suggesting high-energy, assertive emotional episodes."),
    ]

    for dim_a, dim_b, title, description in pairs:
        col_a = f"{dim_a}_prob"
        col_b = f"{dim_b}_prob"
        if col_a not in df.columns or col_b not in df.columns:
            continue
        corr = float(df[col_a].corr(df[col_b]))
        if abs(corr) < corr_threshold:
            continue

        insights.append(_make_insight(
            user_id      = user_id,
            insight_type = InsightType.CORRELATION,
            title        = title,
            description  = (
                f"{description} "
                f"Correlation between {dim_a} and {dim_b}: {corr:.2f}."
            ),
            severity     = "info",
            value        = corr,
            period_start = df["timestamp"].min(),
            period_end   = df["timestamp"].max(),
            tags         = [dim_a, dim_b, "correlation"],
        ))

    return insights


# ── Master insight engine ─────────────────────────────────────────

class InsightEngine:
    """
    Runs all detectors on a user's history and returns a ranked list.

    Usage:
        engine = InsightEngine()
        insights = engine.generate(user_id, history_df)
    """

    def generate(
        self,
        user_id    : str,
        history_df : pd.DataFrame,
        max_insights: int = 10,
    ) -> List[Insight]:
        """
        Run all insight detectors.

        Args:
            user_id    : user identifier
            history_df : DataFrame from PredictionDB.get_user_history()
                         must contain timestamp, valence_prob, arousal_prob,
                         dominance_prob columns
            max_insights: cap on returned insights

        Returns:
            List[Insight] sorted by severity (alert > warning > info)
        """
        if history_df.empty:
            logger.info(f"No history for user {user_id} — skipping insights")
            return []

        # ensure timestamp is datetime
        history_df = history_df.copy()
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])

        all_insights: List[Insight] = []

        detectors = [
            lambda df: detect_trends(df, user_id),
            lambda df: detect_anomalies(df, user_id),
            lambda df: detect_peaks(df, user_id),
            lambda df: detect_stability(df, user_id),
            lambda df: detect_correlations(df, user_id),
        ]

        for detector in detectors:
            try:
                found = detector(history_df)
                all_insights.extend(found)
            except Exception as e:
                logger.warning(f"Insight detector failed: {e}")

        # Sort: alert → warning → info
        severity_order = {"alert": 0, "warning": 1, "info": 2}
        all_insights.sort(key=lambda x: severity_order.get(x.severity, 3))

        logger.info(
            f"Generated {len(all_insights)} insights for user {user_id} "
            f"from {len(history_df)} records"
        )
        return all_insights[:max_insights]

    def update_user_baseline(
        self,
        history_df : pd.DataFrame,
        alpha      : float = 0.1,
    ) -> dict:
        """
        Compute exponential moving average baselines for a user.

        Args:
            history_df: sorted prediction history
            alpha: EMA smoothing factor (0=no update, 1=full replacement)

        Returns:
            {"baseline_valence": float, "baseline_arousal": float,
             "baseline_dominance": float}
        """
        if history_df.empty:
            return {
                "baseline_valence"  : 0.5,
                "baseline_arousal"  : 0.5,
                "baseline_dominance": 0.5,
            }

        df = history_df.sort_values("timestamp")
        baselines = {}

        for dim, col in zip(DIMENSIONS, PROB_COLS):
            if col not in df.columns:
                baselines[f"baseline_{dim}"] = 0.5
                continue
            vals = df[col].values
            ema  = float(vals[0])
            for v in vals[1:]:
                ema = alpha * float(v) + (1 - alpha) * ema
            baselines[f"baseline_{dim}"] = round(ema, 4)

        return baselines