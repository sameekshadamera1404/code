from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from recommender import HybridRecommendationEngine


from db import load_data
from engine import RecommendationService

app = FastAPI()


# ✅ Root endpoint
@app.get("/")
def root():
    return {"message": "Hybrid ML Risk API running"}


# ✅ Load data from DB
df = load_data()

# ✅ Initialize service (ML + risk logic)
service = RecommendationService(df)


# ✅ Request schema
class RequestModel(BaseModel):
    threats: list[str] = []
    controls: list[str] = []


# ✅ Recommendation endpoint
@app.post("/recommend")
def recommend(req: RequestModel):

    threats_input = list(set(req.threats))
    controls_input = list(set(req.controls))

    # ✅ Validate input
    if not threats_input and not controls_input:
        raise HTTPException(status_code=400, detail="Empty input")

    # ===============================
    # ✅ CASE 1: Threat → Controls
    # ===============================
    if threats_input:
        threat_results = {}
        
        for threat in threats_input:
            results = service.get_controls([threat])
            if results:
                threat_results[threat] = {
                    "top3_controls": results[:3],
                    "top5_controls": results[:5]
                }
        
        if not threat_results:
            raise HTTPException(status_code=404, detail="No controls found")

        return {
            "status": "success",
            "message": "Controls recommended for given threats",
            "payload": threat_results
        }

    # ===============================
    # ✅ CASE 2: Control → Threats
    # ===============================
    if controls_input:
        control_results = {}
        
        for control in controls_input:
            results = service.get_threats([control])
            if results:
                control_results[control] = {
                    "top3_threats": results[:3],
                    "top5_threats": results[:5]
                }
        
        if not control_results:
            raise HTTPException(status_code=404, detail="No threats found")

        return {
            "status": "success",
            "message": "Threats recommended for given controls",
            "payload": control_results
        }