"""
Pre-compute predictions locally for Concurso 2321.
Run this on your local machine, then push the resulting
predictions.json and status.json to the repo.
Render will load and serve them directly without fitting the model.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
from datetime import datetime
from pathlib import Path

from progol_scraper import get_demo_fixtures
from data_fetcher import build_training_dataset
from dc_prediction_progol import fit_dixon_coles, generate_predictions

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("PRE-COMPUTING PREDICTIONS — CONCURSO 2321")
print("=" * 60)

# Step 1: Get fixtures
print("\nStep 1: Loading fixtures...")
fixtures_data = get_demo_fixtures()
all_fixtures = fixtures_data["progol"] + fixtures_data["revancha"]
print(f"  {len(fixtures_data['progol'])} Progol + {len(fixtures_data['revancha'])} Revancha matches")

# Step 2: Build training data
print("\nStep 2: Building training dataset...")
training_data = build_training_dataset(fixtures=all_fixtures, force_refresh=True)
print(f"  Dataset: {len(training_data)} matches, {training_data['home'].nunique()} teams")

# Step 3: Fit model
print("\nStep 3: Fitting Dixon-Coles model...")
params = fit_dixon_coles(training_data, xi=0.005)
print(f"  rho={params['rho']:.4f}, home_adv={params['home_advantage']:.4f}, converged={params['convergence']}")

# Step 4: Generate predictions
print("\nStep 4: Generating predictions...")
progol_preds = generate_predictions(fixtures_data["progol"], params)
revancha_preds = generate_predictions(fixtures_data["revancha"], params)

# Step 5: Save to JSON
predictions = {
    "progol": progol_preds,
    "revancha": revancha_preds,
    "concurso": fixtures_data.get("concurso", ""),
    "generated_at": datetime.now().isoformat()
}

status = {
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

pred_file = DATA_DIR / "predictions.json"
status_file = DATA_DIR / "status.json"

with open(pred_file, "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)

with open(status_file, "w", encoding="utf-8") as f:
    json.dump(status, f, ensure_ascii=False, indent=2)

print(f"\n{'=' * 60}")
print(f"✅ PREDICTIONS SAVED!")
print(f"   {pred_file}")
print(f"   {status_file}")
print(f"   Progol: {len(progol_preds)} predictions")
print(f"   Revancha: {len(revancha_preds)} predictions")
print(f"   Concurso: {predictions['concurso']}")
