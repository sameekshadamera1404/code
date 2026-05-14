"""
model_store.py
--------------
Handles persistence of the trained HybridRecommendationEngine so that
knowledge of DB mappings is retained even after those rows are deleted.

Flow:
  1. On first run  → train on full DB snapshot → save to disk
  2. On later runs → load saved model from disk (stale DB rows still "remembered")
  3. On /retrain   → merge current DB into existing model → save again

The saved artefact is a single joblib file at MODEL_PATH.
"""

import os
import logging
from datetime import datetime

import joblib
import pandas as pd

MODEL_PATH = os.environ.get("MODEL_PATH", "model_snapshot.joblib")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_model(engine, metadata: dict | None = None):
    """Persist the trained engine + optional metadata to MODEL_PATH."""
    payload = {
        "engine": engine,
        "saved_at": datetime.utcnow().isoformat(),
        "metadata": metadata or {},
    }
    joblib.dump(payload, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)


def load_model():
    """
    Load and return (engine, metadata) from MODEL_PATH.
    Returns (None, {}) if no saved model exists yet.
    """
    if not os.path.exists(MODEL_PATH):
        logger.info("No saved model found at %s", MODEL_PATH)
        return None, {}

    payload = joblib.load(MODEL_PATH)
    logger.info(
        "Model loaded from %s (saved at %s)",
        MODEL_PATH,
        payload.get("saved_at", "unknown"),
    )
    return payload["engine"], payload.get("metadata", {})


def model_exists() -> bool:
    return os.path.exists(MODEL_PATH)
