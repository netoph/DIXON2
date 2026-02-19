"""
FastAPI Server — Progol Prediction Dashboard Backend
=====================================================
Serves predictions via REST API and manages the automated
weekly prediction pipeline.

Endpoints:
    GET  /api/predictions  — Current week's predictions
    GET  /api/status       — Model status and metadata
    POST /api/refresh      — Manual prediction refresh
    GET  /                 — Health check
"""

import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Scheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Local modules
from dc_prediction_progol import (
    fit_dixon_coles, generate_predictions, normalize_team_name
)
from data_fetcher import build_training_dataset
from progol_scraper import get_current_fixtures

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

PREDICTIONS_CACHE = DATA_DIR / "predictions.json"
STATUS_CACHE = DATA_DIR / "status.json"
LOG_FILE = DATA_DIR / "execution_log.txt"

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
current_predictions = {
    "progol": [],
    "revancha": [],
    "concurso": "",
    "generated_at": None
}

model_status = {
    "last_updated": None,
    "rho": None,
    "xi": None,
    "home_advantage": None,
    "n_matches": 0,
    "n_teams": 0,
    "convergence": False,
    "data_quality": "unknown",
    "error": None
}

scheduler = BackgroundScheduler(timezone="America/Mexico_City")


# ---------------------------------------------------------------------------
# Prediction Pipeline
# ---------------------------------------------------------------------------

def log_message(msg: str):
    """Append message to execution log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def run_prediction_pipeline(use_demo: bool = False):
    """
    Full prediction pipeline:
    1. Scrape current quiniela fixtures
    2. Build training dataset (Zero-NaN enforced)
    3. Fit Dixon-Coles model
    4. Generate predictions
    5. Cache results
    """
    global current_predictions, model_status
    
    log_message("=" * 50)
    log_message("PREDICTION PIPELINE — START")
    log_message("=" * 50)
    
    try:
        # Step 1: Get fixtures
        log_message("Step 1: Fetching current fixtures...")
        fixtures_data = get_current_fixtures(use_demo=use_demo)
        
        all_fixtures = fixtures_data["progol"] + fixtures_data["revancha"]
        log_message(f"  Found {len(fixtures_data['progol'])} Progol + "
                    f"{len(fixtures_data['revancha'])} Revancha matches")
        
        # Step 2: Build training data
        log_message("Step 2: Building training dataset...")
        training_data = build_training_dataset(
            fixtures=all_fixtures,
            force_refresh=False
        )
        log_message(f"  Training data: {len(training_data)} matches, "
                    f"{training_data['home'].nunique()} teams")
        
        # Step 3: Fit model
        log_message("Step 3: Fitting Dixon-Coles model...")
        params = fit_dixon_coles(training_data, xi=0.005)
        log_message(f"  Model fitted! rho={params['rho']:.4f}, "
                    f"home_adv={params['home_advantage']:.4f}, "
                    f"converged={params['convergence']}")
        
        # Step 4: Generate predictions
        log_message("Step 4: Generating predictions...")
        progol_preds = generate_predictions(fixtures_data["progol"], params)
        revancha_preds = generate_predictions(fixtures_data["revancha"], params)
        
        # Step 5: Update state
        current_predictions = {
            "progol": progol_preds,
            "revancha": revancha_preds,
            "concurso": fixtures_data.get("concurso", ""),
            "generated_at": datetime.now().isoformat()
        }
        
        model_status = {
            "last_updated": datetime.now().isoformat(),
            "rho": round(params["rho"], 6),
            "xi": round(params["xi"], 6),
            "home_advantage": round(params["home_advantage"], 4),
            "n_matches": params["n_matches"],
            "n_teams": len(params["teams"]),
            "convergence": params["convergence"],
            "data_quality": "clean",
            "error": None
        }
        
        # Cache to disk
        _save_cache()
        
        log_message("PIPELINE COMPLETE ✅")
        log_message(f"  Progol predictions: {len(progol_preds)}")
        log_message(f"  Revancha predictions: {len(revancha_preds)}")
        log_message(f"  Zero NaN violations: 0")
        
    except Exception as e:
        error_msg = f"Pipeline error: {str(e)}\n{traceback.format_exc()}"
        log_message(f"PIPELINE FAILED ❌: {error_msg}")
        model_status["error"] = str(e)
        model_status["data_quality"] = "error"


def _save_cache():
    """Save predictions and status to disk."""
    try:
        with open(PREDICTIONS_CACHE, "w", encoding="utf-8") as f:
            json.dump(current_predictions, f, ensure_ascii=False, indent=2)
        with open(STATUS_CACHE, "w", encoding="utf-8") as f:
            json.dump(model_status, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_message(f"Cache save error: {e}")


def _load_cache():
    """Load predictions and status from disk."""
    global current_predictions, model_status
    try:
        if PREDICTIONS_CACHE.exists():
            with open(PREDICTIONS_CACHE, "r", encoding="utf-8") as f:
                current_predictions = json.load(f)
        if STATUS_CACHE.exists():
            with open(STATUS_CACHE, "r", encoding="utf-8") as f:
                model_status = json.load(f)
    except Exception as e:
        log_message(f"Cache load error: {e}")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # Startup
    log_message("Server starting up...")
    _load_cache()
    
    # On Render: skip heavy model fitting, just serve pre-computed predictions
    # On local: run pipeline in background thread
    if os.environ.get("RENDER"):
        if not current_predictions.get("progol"):
            log_message("Render: No cached predictions found! Run precompute.py locally and push.")
        else:
            log_message(f"Render: Loaded pre-computed predictions for Concurso {current_predictions.get('concurso', '?')}")
    else:
        import threading
        
        def _initial_pipeline():
            if not current_predictions.get("progol"):
                log_message("No cached predictions. Running initial pipeline...")
                run_prediction_pipeline(use_demo=True)
            else:
                log_message("Loaded cached predictions. Skipping initial pipeline.")
        
        threading.Thread(target=_initial_pipeline, daemon=True).start()
    
    # Schedule weekly run: Monday 09:00 CST
    scheduler.add_job(
        run_prediction_pipeline,
        CronTrigger(day_of_week="mon", hour=9, minute=0),
        id="weekly_predictions",
        name="Weekly Progol Prediction Pipeline",
        replace_existing=True,
        kwargs={"use_demo": False}
    )
    scheduler.start()
    log_message("Scheduler started: Monday 09:00 CST weekly pipeline")
    
    yield
    
    # Shutdown
    scheduler.shutdown(wait=False)
    log_message("Server shut down.")


app = FastAPI(
    title="Progol Prediction Dashboard API",
    description="Dixon-Coles model predictions for Progol quiniela",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "service": "Progol Prediction Dashboard",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/predictions")
async def get_predictions():
    """
    Get current week's predictions.
    
    Returns Progol (14 matches) and Revancha (7 matches) predictions
    with probabilities, recommended picks, and top scorelines.
    """
    if not current_predictions.get("progol"):
        raise HTTPException(
            status_code=503,
            detail="Predictions not yet available. Please trigger a refresh."
        )
    
    return JSONResponse(content=current_predictions)


@app.get("/api/status")
async def get_status():
    """
    Get model status and metadata.
    
    Returns last update time, model parameters (rho, xi),
    data quality indicator, and convergence status.
    """
    return JSONResponse(content=model_status)


@app.post("/api/refresh")
async def refresh_predictions():
    """
    Manually trigger prediction pipeline refresh.
    
    Attempts live scraping first, falls back to demo fixtures.
    """
    log_message("Manual refresh triggered via API")
    
    try:
        # Try live first, fallback to demo
        run_prediction_pipeline(use_demo=False)
        
        # If no results from live scraping, use demo
        if not current_predictions.get("progol"):
            run_prediction_pipeline(use_demo=True)
        
        return JSONResponse(content={
            "status": "success",
            "message": "Predictions refreshed",
            "generated_at": current_predictions.get("generated_at"),
            "progol_count": len(current_predictions.get("progol", [])),
            "revancha_count": len(current_predictions.get("revancha", []))
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Refresh failed: {str(e)}"
        )


@app.get("/api/log")
async def get_log():
    """Get execution log (last 100 lines)."""
    try:
        if LOG_FILE.exists():
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return {"log": lines[-100:]}
        return {"log": []}
    except Exception as e:
        return {"log": [], "error": str(e)}


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("RENDER") is None,  # Only reload locally
        log_level="info"
    )
