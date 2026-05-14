"""
main.py
-------
FastAPI app with persistent ML model.

Startup logic
─────────────
1. Load data from DB.
2. Try to load a previously saved model from disk.
   • Found  → use it as-is (retains knowledge of deleted DB rows).
   • Missing → train fresh on current DB, then save to disk.
3. Wrap the engine in RecommendationService for risk enrichment.

Endpoints
─────────
POST /recommend  — same contract as before
POST /retrain    — merge current DB into the saved model and re-save
GET  /model-info — show when the model was last saved and how many
                   threats / controls it knows about
"""

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from db import load_data
from engine import RecommendationService
from model_store import load_model, save_model, model_exists
from recommender import HybridRecommendationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


# ---------------------------------------------------------------------------
# Startup: load or train model
# ---------------------------------------------------------------------------

df = load_data()

_saved_engine, _metadata = load_model()

if _saved_engine is not None:
    # ── Use persisted model ─────────────────────────────────────────────
    logger.info("Using persisted model (saved at %s)", _metadata.get("saved_at", "?"))
    ml_engine = _saved_engine
else:
    # ── First run: train from scratch and persist ───────────────────────
    logger.info("No saved model found — training from scratch on current DB...")
    ml_engine = HybridRecommendationEngine()
    ml_engine.fit(df)
    save_model(ml_engine, metadata={"trained_on_rows": len(df)})
    logger.info("Model trained and saved.")

service = RecommendationService(df, engine=ml_engine)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class RequestModel(BaseModel):
    threats: list[str] = []
    controls: list[str] = []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "Hybrid ML Risk API running"}


@app.post("/recommend")
def recommend(req: RequestModel):
    threats_input = list(set(req.threats))
    controls_input = list(set(req.controls))

    if not threats_input and not controls_input:
        raise HTTPException(status_code=400, detail="Empty input")

    # ── Threat → Controls ──────────────────────────────────────────────
    if threats_input:
        threat_results = {}
        for threat in threats_input:
            results = service.get_controls([threat])
            if results:
                threat_results[threat] = {
                    "top3_controls": results[:3],
                    "top5_controls": results[:5],
                }
        if not threat_results:
            raise HTTPException(status_code=404, detail="No controls found")
        return {
            "status": "success",
            "message": "Controls recommended for given threats",
            "payload": threat_results,
        }

    # ── Control → Threats ──────────────────────────────────────────────
    if controls_input:
        control_results = {}
        for control in controls_input:
            results = service.get_threats([control])
            if results:
                control_results[control] = {
                    "top3_threats": results[:3],
                    "top5_threats": results[:5],
                }
        if not control_results:
            raise HTTPException(status_code=404, detail="No threats found")
        return {
            "status": "success",
            "message": "Threats recommended for given controls",
            "payload": control_results,
        }


@app.post("/retrain")
def retrain():
    """
    Pull current DB state and merge it into the saved model.

    • Rows that exist in DB → re-enforced in the model.
    • Rows deleted from DB  → still remembered from previous snapshot.
    • Brand-new rows        → added to model knowledge.
    """
    global df, service, ml_engine

    logger.info("Retraining requested — loading current DB snapshot...")
    new_df = load_data()

    ml_engine.merge_fit(new_df)
    save_model(ml_engine, metadata={"trained_on_rows": len(new_df)})

    # Rebuild service with fresh DB data (for direct-lookup risk enrichment)
    df = new_df
    service = RecommendationService(df, engine=ml_engine)

    logger.info("Retrain complete. Model re-saved.")
    return {
        "status": "success",
        "message": "Model retrained and saved. Deleted mappings are still remembered.",
        "db_rows_used": len(new_df),
        "threats_known": len(ml_engine.threats),
        "controls_known": len(ml_engine.controls),
    }


@app.get("/model-info")
def model_info():
    """Return metadata about the currently loaded model."""
    _, meta = load_model()
    return {
        "threats_known": len(ml_engine.threats),
        "controls_known": len(ml_engine.controls),
        "last_saved_at": meta.get("saved_at", "unknown"),
        "snapshot_rows": (
            len(ml_engine._training_snapshot)
            if ml_engine._training_snapshot is not None
            else None
        ),
    }
