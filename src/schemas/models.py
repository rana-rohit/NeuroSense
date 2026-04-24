"""
schemas/models.py
Pydantic data contracts shared across API, pipeline, storage, and insights.
Single source of truth for all data shapes.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import uuid

try:
    from pydantic import BaseModel, Field, field_validator
    _PYDANTIC = True
except ImportError:
    # Lightweight fallback for environments without pydantic
    # Full validation only available with pydantic installed
    import dataclasses

    class Field:  # noqa: N801
        def __init__(self, default=None, default_factory=None, **kwargs):
            self._default = default
            self._factory = default_factory
        def __call__(self):
            return self._factory() if self._factory else self._default

    class _Meta(type):
        """Metaclass: auto-apply default_factory for Field() annotations."""
        def __call__(cls, **kwargs):
            obj = object.__new__(cls)
            hints = {}
            for klass in reversed(cls.__mro__):
                hints.update(getattr(klass, '__annotations__', {}))
            for name, _ in hints.items():
                default = getattr(cls, name, None)
                if isinstance(default, Field):
                    kwargs.setdefault(name, default())
                elif callable(default):
                    pass
            for k, v in kwargs.items():
                setattr(obj, k, v)
            return obj

    class BaseModel(metaclass=_Meta):
        def model_dump(self):
            return {k: v for k, v in self.__dict__.items()}

    def field_validator(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    _PYDANTIC = False


# ── Enums ─────────────────────────────────────────────────────────

class EmotionDimension(str, Enum):
    VALENCE   = "valence"
    AROUSAL   = "arousal"
    DOMINANCE = "dominance"

class EmotionLabel(str, Enum):
    HIGH = "High"
    LOW  = "Low"

class SignalQuality(str, Enum):
    GOOD      = "good"
    DEGRADED  = "degraded"
    POOR      = "poor"

class InsightType(str, Enum):
    TREND         = "trend"
    ANOMALY       = "anomaly"
    PEAK          = "peak"
    STABILITY     = "stability"
    CORRELATION   = "correlation"


# ── Signal input ──────────────────────────────────────────────────

class SignalInput(BaseModel):
    """Raw EEG + ECG signal payload from a device or upload."""

    session_id    : str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id       : str
    timestamp     : datetime = Field(default_factory=datetime.utcnow)

    # Flat lists — shape validated in pipeline
    eeg_data      : List[List[float]] = Field(
        description="EEG samples: [[ch0..ch13], ...] shape (N, 14)"
    )
    ecg_data      : List[List[float]] = Field(
        description="ECG samples: [[ch0, ch1], ...] shape (M, 2)"
    )
    eeg_fs        : float = Field(default=128.0, description="EEG sampling rate Hz")
    ecg_fs        : float = Field(default=256.0, description="ECG sampling rate Hz")

    # Optional baseline (if device sends it)
    eeg_baseline  : Optional[List[List[float]]] = None
    ecg_baseline  : Optional[List[List[float]]] = None

    metadata      : Dict[str, str] = Field(default_factory=dict)

    @field_validator("eeg_data")
    @classmethod
    def validate_eeg_channels(cls, v):
        if v and len(v[0]) != 14:
            raise ValueError(
                f"EEG must have 14 channels, got {len(v[0])}"
            )
        if len(v) < 128 * 5:  # minimum 5 seconds
            raise ValueError(
                f"EEG too short: {len(v)} samples (need >= {128*5})"
            )
        return v

    @field_validator("ecg_data")
    @classmethod
    def validate_ecg_channels(cls, v):
        if v and len(v[0]) != 2:
            raise ValueError(
                f"ECG must have 2 channels, got {len(v[0])}"
            )
        return v


# ── Window-level prediction ───────────────────────────────────────

class WindowPrediction(BaseModel):
    """Prediction for a single time window."""
    window_index  : int
    dimension     : EmotionDimension
    label         : EmotionLabel
    prob_high     : float = Field(ge=0.0, le=1.0)
    prob_low      : float = Field(ge=0.0, le=1.0)
    confidence    : float = Field(ge=0.0, le=1.0)


# ── Session-level prediction ──────────────────────────────────────

class EmotionPrediction(BaseModel):
    """Aggregated prediction for one full signal session."""
    prediction_id : str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id    : str
    user_id       : str
    timestamp     : datetime = Field(default_factory=datetime.utcnow)

    # Aggregate results per dimension
    valence       : EmotionLabel
    arousal       : EmotionLabel
    dominance     : EmotionLabel

    # Confidence scores
    valence_conf  : float = Field(ge=0.0, le=1.0)
    arousal_conf  : float = Field(ge=0.0, le=1.0)
    dominance_conf: float = Field(ge=0.0, le=1.0)

    # Probability of High class per dimension
    valence_prob  : float = Field(ge=0.0, le=1.0)
    arousal_prob  : float = Field(ge=0.0, le=1.0)
    dominance_prob: float = Field(ge=0.0, le=1.0)

    # Window-level detail
    n_windows     : int
    window_preds  : List[WindowPrediction] = Field(default_factory=list)

    # Signal quality
    signal_quality: SignalQuality = SignalQuality.GOOD
    quality_notes : str = ""

    # Processing metadata
    model_version : str = "1.0.0"
    processing_ms : float = 0.0


# ── Storage record ────────────────────────────────────────────────

class PredictionRecord(BaseModel):
    """Flattened record stored in the database / parquet."""
    record_id      : str = Field(default_factory=lambda: str(uuid.uuid4()))
    prediction_id  : str
    session_id     : str
    user_id        : str
    timestamp      : datetime

    valence        : str   # High | Low
    arousal        : str
    dominance      : str

    valence_prob   : float
    arousal_prob   : float
    dominance_prob : float

    n_windows      : int
    signal_quality : str
    model_version  : str
    processing_ms  : float


# ── Insight ───────────────────────────────────────────────────────

class Insight(BaseModel):
    """A single derived insight from historical predictions."""
    insight_id    : str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id       : str
    generated_at  : datetime = Field(default_factory=datetime.utcnow)

    insight_type  : InsightType
    dimension     : Optional[EmotionDimension] = None
    title         : str
    description   : str
    severity      : str = "info"   # info | warning | alert

    # Supporting data
    value         : Optional[float] = None
    reference     : Optional[float] = None
    period_start  : Optional[datetime] = None
    period_end    : Optional[datetime] = None

    tags          : List[str] = Field(default_factory=list)


# ── User profile ──────────────────────────────────────────────────

class UserProfile(BaseModel):
    """Persistent user profile with emotional baseline."""
    user_id       : str
    created_at    : datetime = Field(default_factory=datetime.utcnow)
    updated_at    : datetime = Field(default_factory=datetime.utcnow)

    # Rolling baseline probabilities (updated per session)
    baseline_valence   : float = 0.5
    baseline_arousal   : float = 0.5
    baseline_dominance : float = 0.5

    total_sessions     : int = 0
    total_windows      : int = 0

    metadata           : Dict[str, str] = Field(default_factory=dict)


# ── API responses ─────────────────────────────────────────────────

class PredictResponse(BaseModel):
    status        : str = "success"
    prediction    : EmotionPrediction
    insights      : List[Insight] = Field(default_factory=list)
    message       : str = ""

class HistoryResponse(BaseModel):
    user_id       : str
    total_records : int
    records       : List[PredictionRecord]
    page          : int = 1
    page_size     : int = 50

class InsightResponse(BaseModel):
    user_id       : str
    period_days   : int
    insights      : List[Insight]
    generated_at  : datetime = Field(default_factory=datetime.utcnow)

class HealthResponse(BaseModel):
    status        : str
    version       : str
    model_loaded  : bool
    db_connected  : bool
    uptime_seconds: float