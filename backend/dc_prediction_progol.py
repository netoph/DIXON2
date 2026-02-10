"""
Dixon-Coles Prediction Engine for Progol
==========================================
Implementation of the Dixon & Coles (1997) model for predicting football match
outcomes. Used as the core prediction engine for weekly Progol quiniela forecasting.

Key features:
- Rho correction for low-scoring matches (0-0, 0-1, 1-0, 1-1)
- Time-decay weighting (xi parameter) to prioritize recent form
- Full scoreline probability matrix generation
- Aggregated match outcome probabilities (Home/Draw/Away)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from datetime import datetime, timedelta
from unidecode import unidecode
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_GOALS = 10  # Max goals per team in probability matrix
DEFAULT_XI = 0.005  # Default time-decay parameter


# ---------------------------------------------------------------------------
# Team Name Normalization
# ---------------------------------------------------------------------------
TEAM_ALIASES = {
    # Liga MX
    "club america": "america",
    "cf america": "america",
    "club de futbol america": "america",
    "aguilas": "america",
    "aguilas del america": "america",
    "guadalajara": "chivas",
    "cd guadalajara": "chivas",
    "club deportivo guadalajara": "chivas",
    "chivas de guadalajara": "chivas",
    "chivas guadalajara": "chivas",
    "unam": "pumas",
    "pumas unam": "pumas",
    "club universidad nacional": "pumas",
    "universidad nacional": "pumas",
    "cruz azul": "cruz azul",
    "cf monterrey": "monterrey",
    "club de futbol monterrey": "monterrey",
    "rayados": "monterrey",
    "rayados de monterrey": "monterrey",
    "tigres uanl": "tigres",
    "cf tigres": "tigres",
    "club tigres": "tigres",
    "santos laguna": "santos",
    "club santos laguna": "santos",
    "deportivo toluca": "toluca",
    "toluca fc": "toluca",
    "club toluca": "toluca",
    "diablos rojos": "toluca",
    "leon": "leon",
    "club leon": "leon",
    "la fiera": "leon",
    "atlas": "atlas",
    "atlas fc": "atlas",
    "atlas guadalajara": "atlas",
    "puebla": "puebla",
    "club puebla": "puebla",
    "puebla fc": "puebla",
    "la franja": "puebla",
    "pachuca": "pachuca",
    "cf pachuca": "pachuca",
    "club pachuca": "pachuca",
    "tuzos": "pachuca",
    "necaxa": "necaxa",
    "club necaxa": "necaxa",
    "rayos del necaxa": "necaxa",
    "queretaro": "queretaro",
    "club queretaro": "queretaro",
    "queretaro fc": "queretaro",
    "gallos blancos": "queretaro",
    "tijuana": "tijuana",
    "club tijuana": "tijuana",
    "xolos": "tijuana",
    "xolos de tijuana": "tijuana",
    "mazatlan": "mazatlan",
    "mazatlan fc": "mazatlan",
    "juarez": "juarez",
    "fc juarez": "juarez",
    "bravos": "juarez",
    "bravos de juarez": "juarez",
    "san luis": "san luis",
    "atletico san luis": "san luis",
    "atletico de san luis": "san luis",
    # International common aliases
    "manchester united": "man united",
    "manchester city": "man city",
    "tottenham hotspur": "tottenham",
    "tottenham": "tottenham",
    "wolverhampton wanderers": "wolves",
    "wolverhampton": "wolves",
    "newcastle united": "newcastle",
    "west ham united": "west ham",
    "brighton and hove albion": "brighton",
    "brighton & hove albion": "brighton",
    "nottingham forest": "nott'ham forest",
    "atletico madrid": "atletico madrid",
    "atletico de madrid": "atletico madrid",
    "real sociedad": "real sociedad",
    "real betis": "real betis",
    "paris saint-germain": "psg",
    "paris saint germain": "psg",
    "paris sg": "psg",
    "bayern munich": "bayern munich",
    "bayern munchen": "bayern munich",
    "bayern de munich": "bayern munich",
    "fc bayern": "bayern munich",
    "borussia dortmund": "dortmund",
    "bayer leverkusen": "leverkusen",
    "bayer 04 leverkusen": "leverkusen",
    "rb leipzig": "rb leipzig",
    "inter milan": "inter",
    "internazionale": "inter",
    "fc internazionale": "inter",
    "ac milan": "milan",
    "juventus": "juventus",
    "juventus fc": "juventus",
}


def normalize_team_name(name: str) -> str:
    """Normalize a team name using unidecode and alias lookup."""
    if not isinstance(name, str):
        return str(name)
    cleaned = unidecode(name).strip().lower()
    # Remove common prefixes/suffixes
    for prefix in ["club ", "cf ", "fc ", "cd ", "c.f. "]:
        if cleaned.startswith(prefix) and cleaned != prefix.strip():
            pass  # Keep full name for alias lookup
    return TEAM_ALIASES.get(cleaned, cleaned)


# ---------------------------------------------------------------------------
# Dixon-Coles Model Core
# ---------------------------------------------------------------------------

def rho_correction(x: int, y: int, lambda_: float, mu: float, rho: float) -> float:
    """
    Apply the Dixon-Coles rho correction for low-scoring matches.
    
    This adjusts the independent Poisson assumption for scores:
    (0,0), (1,0), (0,1), (1,1) where correlation is empirically observed.
    
    Args:
        x: Home team goals
        y: Away team goals
        lambda_: Expected home goals (Poisson rate)
        mu: Expected away goals (Poisson rate)
        rho: Correlation parameter (typically small, near 0)
    
    Returns:
        Multiplicative correction factor tau(x, y, lambda, mu, rho)
    """
    if x == 0 and y == 0:
        return 1.0 - lambda_ * mu * rho
    elif x == 0 and y == 1:
        return 1.0 + lambda_ * rho
    elif x == 1 and y == 0:
        return 1.0 + mu * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0


def _time_weight(dates: pd.Series, xi: float) -> np.ndarray:
    """
    Compute exponential time-decay weights.
    
    More recent matches receive higher weight. The decay rate is
    controlled by xi: higher xi = faster decay = more emphasis on recent form.
    
    Args:
        dates: Series of match dates
        xi: Decay rate parameter
    
    Returns:
        Array of weights in [0, 1]
    """
    if dates is None or len(dates) == 0:
        return np.ones(0)
    
    max_date = dates.max()
    days_diff = (max_date - dates).dt.days.values.astype(float)
    weights = np.exp(-xi * days_diff)
    return weights


def neg_log_likelihood(params: np.ndarray, home_idx: np.ndarray,
                       away_idx: np.ndarray, home_goals: np.ndarray,
                       away_goals: np.ndarray, weights: np.ndarray,
                       n_teams: int) -> float:
    """
    Vectorized negative log-likelihood for the Dixon-Coles model.
    
    Uses pre-computed integer indices for teams instead of dict lookups.
    All operations are vectorized with numpy for fast execution.
    
    Args:
        params: Flat parameter vector [attack(n), defense(n), gamma, rho]
        home_idx: Integer indices for home teams
        away_idx: Integer indices for away teams
        home_goals: Array of home goals scored
        away_goals: Array of away goals scored
        weights: Time-decay weights
        n_teams: Number of unique teams
    
    Returns:
        Negative log-likelihood (to be minimized)
    """
    attack = params[:n_teams]
    defense = params[n_teams:2*n_teams]
    gamma = params[2*n_teams]       # home advantage
    rho = params[2*n_teams + 1]     # correlation parameter
    
    # Expected goals (vectorized)
    lambda_ = np.exp(attack[home_idx] + defense[away_idx] + gamma)
    mu = np.exp(attack[away_idx] + defense[home_idx])
    
    # Clamp
    lambda_ = np.maximum(lambda_, 1e-6)
    mu = np.maximum(mu, 1e-6)
    
    # Poisson log-PMF (vectorized): k*ln(λ) - λ - ln(k!)
    log_p_home = home_goals * np.log(lambda_) - lambda_ - _log_factorial(home_goals)
    log_p_away = away_goals * np.log(mu) - mu - _log_factorial(away_goals)
    
    # Rho correction (vectorized)
    tau = np.ones(len(home_goals))
    
    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)
    
    tau[m00] = 1.0 - lambda_[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lambda_[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    
    # Avoid log(0) or log(negative)
    tau = np.maximum(tau, 1e-10)
    
    # Weighted log-likelihood
    log_lik = weights * (log_p_home + log_p_away + np.log(tau))
    
    return -np.sum(log_lik)


# Pre-computed log factorial lookup table (goals rarely exceed 10)
_LOG_FACT_TABLE = np.array([np.sum(np.log(np.arange(1, k+1))) if k > 0 else 0.0 
                             for k in range(20)])

def _log_factorial(k: np.ndarray) -> np.ndarray:
    """Vectorized log(k!) using lookup table."""
    k = np.clip(k.astype(int), 0, len(_LOG_FACT_TABLE) - 1)
    return _LOG_FACT_TABLE[k]


def fit_dixon_coles(data: pd.DataFrame, xi: float = DEFAULT_XI) -> dict:
    """
    Fit the Dixon-Coles model to historical match data.
    
    Uses vectorized log-likelihood for fast optimization even with
    thousands of matches.
    
    Args:
        data: DataFrame with columns [home, away, home_goals, away_goals, date]
              All values must be non-NaN (Zero-NaN Policy).
        xi: Time-decay parameter (default 0.005)
    
    Returns:
        Dictionary with keys:
            - 'attack': {team: attack_rating}
            - 'defense': {team: defense_rating}
            - 'home_advantage': float
            - 'rho': float
            - 'xi': float
            - 'teams': list of team names
            - 'n_matches': int
            - 'convergence': bool
            - 'fitted_at': ISO timestamp
    """
    # Validate Zero-NaN Policy
    critical_cols = ["home", "away", "home_goals", "away_goals"]
    for col in critical_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
        if data[col].isna().any():
            nan_count = data[col].isna().sum()
            raise ValueError(
                f"Zero-NaN Policy violated: {nan_count} NaN values in '{col}'. "
                f"Clean data before fitting."
            )
    
    # Normalize team names
    data = data.copy()
    data["home"] = data["home"].apply(normalize_team_name)
    data["away"] = data["away"].apply(normalize_team_name)
    
    # Parse dates if present
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"])
    
    # Get unique teams and create index mapping
    teams = sorted(list(set(data["home"].tolist() + data["away"].tolist())))
    n = len(teams)
    team_to_idx = {team: i for i, team in enumerate(teams)}
    
    if n < 2:
        raise ValueError(f"Need at least 2 teams, got {n}")
    
    print(f"[Dixon-Coles] Fitting model with {len(data)} matches, {n} teams, xi={xi}")
    
    # Pre-convert to numpy arrays for vectorized computation
    home_idx = data["home"].map(team_to_idx).values.astype(int)
    away_idx = data["away"].map(team_to_idx).values.astype(int)
    home_goals = data["home_goals"].values.astype(int)
    away_goals = data["away_goals"].values.astype(int)
    
    # Time weights
    if "date" in data.columns:
        weights = _time_weight(data["date"], xi)
    else:
        weights = np.ones(len(data))
    
    # Initial parameters: attack=0, defense=0, gamma=0.25, rho=-0.1
    x0 = np.zeros(2*n + 2)
    x0[2*n] = 0.25      # home advantage initial guess
    x0[2*n + 1] = -0.1  # rho initial guess
    
    # Constraint: sum of attack params = 0 (identifiability)
    constraints = [{
        "type": "eq",
        "fun": lambda p, n=n: np.sum(p[:n])
    }]
    
    # Bounds for rho: must keep tau > 0
    bounds = [(None, None)] * (2*n)  # attack & defense: unbounded
    bounds.append((None, None))       # gamma: unbounded
    bounds.append((-1.5, 1.5))        # rho: bounded
    
    # Optimize with vectorized likelihood
    result = minimize(
        neg_log_likelihood,
        x0,
        args=(home_idx, away_idx, home_goals, away_goals, weights, n),
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-8}
    )
    
    attack = dict(zip(teams, result.x[:n]))
    defense = dict(zip(teams, result.x[n:2*n]))
    gamma = result.x[2*n]
    rho = result.x[2*n + 1]
    
    print(f"[Dixon-Coles] Converged: {result.success} | rho={rho:.4f} | "
          f"home_adv={gamma:.4f} | iterations={result.nit}")
    
    return {
        "attack": attack,
        "defense": defense,
        "home_advantage": float(gamma),
        "rho": float(rho),
        "xi": float(xi),
        "teams": teams,
        "n_matches": len(data),
        "convergence": bool(result.success),
        "fitted_at": datetime.now().isoformat()
    }


def predict_match(home: str, away: str, params: dict) -> np.ndarray:
    """
    Generate full scoreline probability matrix for a match.
    
    Args:
        home: Home team name
        away: Away team name
        params: Fitted model parameters from fit_dixon_coles()
    
    Returns:
        (MAX_GOALS+1) x (MAX_GOALS+1) matrix where entry [i,j] = P(home=i, away=j)
    """
    home = normalize_team_name(home)
    away = normalize_team_name(away)
    
    attack = params["attack"]
    defense = params["defense"]
    gamma = params["home_advantage"]
    rho = params["rho"]
    
    # Handle unknown teams with league-average parameters
    avg_attack = np.mean(list(attack.values()))
    avg_defense = np.mean(list(defense.values()))
    
    home_att = attack.get(home, avg_attack)
    home_def = defense.get(home, avg_defense)
    away_att = attack.get(away, avg_attack)
    away_def = defense.get(away, avg_defense)
    
    # Expected goals
    lambda_ = np.exp(home_att + away_def + gamma)
    mu = np.exp(away_att + home_def)
    
    lambda_ = max(lambda_, 1e-6)
    mu = max(mu, 1e-6)
    
    # Build probability matrix
    matrix = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = poisson.pmf(i, lambda_) * poisson.pmf(j, mu)
            tau = rho_correction(i, j, lambda_, mu, rho)
            matrix[i, j] = p * tau
    
    # Normalize to ensure probabilities sum to 1
    total = matrix.sum()
    if total > 0:
        matrix /= total
    
    return matrix


def get_match_probabilities(home: str, away: str, params: dict) -> dict:
    """
    Get aggregated match outcome probabilities.
    
    Args:
        home: Home team name
        away: Away team name
        params: Fitted model parameters
    
    Returns:
        dict with keys: 'home_win', 'draw', 'away_win' (probabilities summing to ~1)
    """
    matrix = predict_match(home, away, params)
    
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            if i > j:
                home_win += matrix[i, j]
            elif i == j:
                draw += matrix[i, j]
            else:
                away_win += matrix[i, j]
    
    return {
        "home_win": round(float(home_win), 4),
        "draw": round(float(draw), 4),
        "away_win": round(float(away_win), 4)
    }


def get_top_scorelines(home: str, away: str, params: dict, n: int = 3) -> list:
    """
    Get the n most probable scorelines for a match.
    
    Args:
        home: Home team name
        away: Away team name
        params: Fitted model parameters
        n: Number of top scorelines to return
    
    Returns:
        List of dicts: [{'home_goals': int, 'away_goals': int, 'probability': float}, ...]
    """
    matrix = predict_match(home, away, params)
    
    # Flatten and get top n indices
    flat = matrix.flatten()
    top_indices = flat.argsort()[::-1][:n]
    
    results = []
    for idx in top_indices:
        i = idx // (MAX_GOALS + 1)
        j = idx % (MAX_GOALS + 1)
        results.append({
            "home_goals": int(i),
            "away_goals": int(j),
            "probability": round(float(matrix[i, j]), 4)
        })
    
    return results


def get_pick(probs: dict) -> str:
    """
    Determine the recommended pick (L/E/V) based on probabilities.
    
    L = Local (home win), E = Empate (draw), V = Visitante (away win)
    """
    max_key = max(probs, key=probs.get)
    pick_map = {"home_win": "L", "draw": "E", "away_win": "V"}
    return pick_map[max_key]


def standardize_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize a raw schedule DataFrame to the required schema.
    
    Expected output columns:
        - home: str (normalized team name)
        - away: str (normalized team name)
        - home_goals: int
        - away_goals: int
        - date: datetime
    
    Handles common column name variants from different data sources.
    """
    df = df.copy()
    
    # Common column mappings
    col_maps = {
        "home": ["home", "home_team", "hometeam", "local", "equipo_local",
                 "HomeTeam", "Home", "team_home"],
        "away": ["away", "away_team", "awayteam", "visitor", "visitante",
                 "equipo_visitante", "AwayTeam", "Away", "team_away"],
        "home_goals": ["home_goals", "homegoals", "hg", "fthg", "home_score",
                       "goles_local", "HomeGoals", "GF", "score_home", "FTHG"],
        "away_goals": ["away_goals", "awaygoals", "ag", "ftag", "away_score",
                       "goles_visitante", "AwayGoals", "GA", "score_away", "FTAG"],
        "date": ["date", "Date", "fecha", "match_date", "datetime",
                 "matchdate", "game_date"]
    }
    
    renamed = {}
    for target, candidates in col_maps.items():
        for c in candidates:
            if c in df.columns and target not in renamed.values():
                renamed[c] = target
                break
    
    df = df.rename(columns=renamed)
    
    # Validate required columns
    required = ["home", "away", "home_goals", "away_goals"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after standardization: {missing}")
    
    # Clean data
    df["home"] = df["home"].apply(normalize_team_name)
    df["away"] = df["away"].apply(normalize_team_name)
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.Timestamp.now()
    
    # Zero-NaN enforcement
    before = len(df)
    df = df.dropna(subset=required)
    after = len(df)
    if before != after:
        print(f"[Warning] Dropped {before - after} rows with NaN values (Zero-NaN Policy)")
    
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    
    return df[["home", "away", "home_goals", "away_goals", "date"]].reset_index(drop=True)


def generate_predictions(fixtures: list, params: dict) -> list:
    """
    Generate full predictions for a list of fixtures.
    
    Args:
        fixtures: List of dicts with keys: match_num, home, away, league (optional)
        params: Fitted Dixon-Coles parameters
    
    Returns:
        List of prediction dicts with probabilities, pick, and top scorelines
    """
    predictions = []
    
    for fix in fixtures:
        home = fix["home"]
        away = fix["away"]
        
        probs = get_match_probabilities(home, away, params)
        pick = get_pick(probs)
        top_scores = get_top_scorelines(home, away, params, n=3)
        
        predictions.append({
            "match_num": fix.get("match_num", len(predictions) + 1),
            "home": home,
            "away": away,
            "league": fix.get("league", ""),
            "home_win": probs["home_win"],
            "draw": probs["draw"],
            "away_win": probs["away_win"],
            "pick": pick,
            "top_scorelines": top_scores,
            "confidence": max(probs.values())
        })
    
    return predictions


# ---------------------------------------------------------------------------
# Demo / Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    teams = ["america", "chivas", "cruz azul", "pumas", "monterrey",
             "tigres", "santos", "toluca", "leon", "atlas",
             "pachuca", "puebla", "necaxa", "queretaro", "tijuana",
             "mazatlan", "juarez", "san luis"]
    
    n_matches = 300
    rows = []
    base_date = datetime(2025, 1, 1)
    
    for i in range(n_matches):
        home, away = np.random.choice(teams, 2, replace=False)
        hg = np.random.poisson(1.4)
        ag = np.random.poisson(1.1)
        date = base_date + timedelta(days=int(i * 1.5))
        rows.append({"home": home, "away": away, "home_goals": hg,
                      "away_goals": ag, "date": date})
    
    sample_data = pd.DataFrame(rows)
    
    print("=" * 60)
    print("DIXON-COLES MODEL — TEST RUN")
    print("=" * 60)
    
    # Fit model
    params = fit_dixon_coles(sample_data, xi=0.005)
    print(f"\nModel fitted successfully!")
    print(f"  Teams: {len(params['teams'])}")
    print(f"  Matches: {params['n_matches']}")
    print(f"  rho: {params['rho']:.4f}")
    print(f"  Home advantage: {params['home_advantage']:.4f}")
    
    # Test prediction
    print(f"\n--- Sample Prediction: America vs Chivas ---")
    probs = get_match_probabilities("america", "chivas", params)
    print(f"  Home win: {probs['home_win']:.1%}")
    print(f"  Draw:     {probs['draw']:.1%}")
    print(f"  Away win: {probs['away_win']:.1%}")
    print(f"  Pick:     {get_pick(probs)}")
    
    top = get_top_scorelines("america", "chivas", params, n=3)
    print(f"\n  Top scorelines:")
    for s in top:
        print(f"    {s['home_goals']}-{s['away_goals']}  ({s['probability']:.1%})")
    
    print("\n✅ All tests passed!")
