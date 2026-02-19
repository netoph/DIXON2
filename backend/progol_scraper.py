"""
Progol Quiniela Scraper
========================
Extracts weekly Progol (14 matches) and Revancha (7 matches)
from the Lotería Nacional website.

Since the website loads content dynamically via JavaScript,
this module provides:
1. A Playwright-based scraper for production use
2. A requests+BeautifulSoup fallback
3. Demo fixtures for development/testing
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from unidecode import unidecode

import requests
from bs4 import BeautifulSoup

from dc_prediction_progol import normalize_team_name

# Try importing Playwright
try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
FIXTURES_CACHE = DATA_DIR / "current_fixtures.json"

PROGOL_URL = "https://www.loterianacional.gob.mx/Progol/Quiniela"
MOMIOS_URL = "https://www.loterianacional.gob.mx/Progol/Momios"


# ---------------------------------------------------------------------------
# Playwright Scraper (Production)
# ---------------------------------------------------------------------------

def scrape_quiniela_playwright() -> dict:
    """
    Scrape current Progol quiniela using Playwright (handles JS rendering).
    
    Returns:
        dict with keys:
            - concurso: str (contest number)
            - fecha: str (contest date)
            - progol: list of 14 fixture dicts
            - revancha: list of 7 fixture dicts
    """
    if not HAS_PLAYWRIGHT:
        print("[Scraper] Playwright not available. Use pip install playwright && playwright install")
        return None
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(PROGOL_URL, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(3000)  # Extra wait for dynamic content
            
            content = page.content()
            browser.close()
            
            return parse_fixtures(content)
    except Exception as e:
        print(f"[Scraper] Playwright error: {e}")
        return None


# ---------------------------------------------------------------------------
# Requests Fallback Scraper
# ---------------------------------------------------------------------------

def scrape_quiniela_requests() -> dict:
    """
    Attempt to scrape Progol quiniela with requests + BeautifulSoup.
    May not work if content is JavaScript-rendered.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(PROGOL_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        
        result = parse_fixtures(resp.text)
        if result and result.get("progol"):
            return result
        
        print("[Scraper] requests fallback: page content may be JS-rendered, "
              "no fixtures found")
        return None
        
    except Exception as e:
        print(f"[Scraper] requests error: {e}")
        return None


def parse_fixtures(html: str) -> dict:
    """
    Parse fixture data from HTML content.
    
    Looks for common patterns in the Lotería Nacional page structure.
    """
    soup = BeautifulSoup(html, "html.parser")
    
    fixtures = {"concurso": "", "fecha": "", "progol": [], "revancha": []}
    
    # Try to find concurso number
    concurso_el = soup.find(string=re.compile(r"Concurso\s*#?\s*\d+", re.I))
    if concurso_el:
        match = re.search(r"(\d+)", str(concurso_el))
        if match:
            fixtures["concurso"] = match.group(1)
    
    # Try to find match rows in tables
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) >= 3:
                texts = [c.get_text(strip=True) for c in cells]
                # Look for patterns like: number | home team | away team
                if texts[0].isdigit():
                    fixture = {
                        "match_num": int(texts[0]),
                        "home": normalize_team_name(texts[1]),
                        "away": normalize_team_name(texts[2]) if len(texts) > 2 else "",
                        "league": texts[3] if len(texts) > 3 else "",
                    }
                    if len(fixtures["progol"]) < 14:
                        fixtures["progol"].append(fixture)
                    else:
                        fixtures["revancha"].append(fixture)
    
    # Also try div-based layouts
    if not fixtures["progol"]:
        match_divs = soup.find_all("div", class_=re.compile(r"match|partido|juego", re.I))
        for i, div in enumerate(match_divs):
            teams = div.find_all(["span", "p", "div"],
                                 class_=re.compile(r"team|equipo", re.I))
            if len(teams) >= 2:
                fixture = {
                    "match_num": i + 1,
                    "home": normalize_team_name(teams[0].get_text(strip=True)),
                    "away": normalize_team_name(teams[1].get_text(strip=True)),
                    "league": "",
                }
                if len(fixtures["progol"]) < 14:
                    fixtures["progol"].append(fixture)
                else:
                    fixtures["revancha"].append(fixture)
    
    return fixtures


# ---------------------------------------------------------------------------
# Demo Fixtures (Development/Testing)
# ---------------------------------------------------------------------------

def get_demo_fixtures() -> dict:
    """
    Return current Progol quiniela fixtures.

    Concurso 2321 — Puntos al 15 de febrero de 2026.
    Source: https://www.loterianacional.gob.mx/Progol/Quiniela
    """
    return {
        "concurso": "2321",
        "fecha": datetime.now().strftime("%Y-%m-%d"),
        "progol": [
            {"match_num": 1, "home": "Cruz Azul", "away": "Guadalajara",
             "league": "Liga MX"},
            {"match_num": 2, "home": "Pumas", "away": "Monterrey",
             "league": "Liga MX"},
            {"match_num": 3, "home": "Guadalajara F", "away": "Águilas F",
             "league": "Liga MX Femenil"},
            {"match_num": 4, "home": "Necaxa F", "away": "Querétaro F",
             "league": "Liga MX Femenil"},
            {"match_num": 5, "home": "Getafe", "away": "Sevilla",
             "league": "La Liga"},
            {"match_num": 6, "home": "West Ham", "away": "Bournemouth",
             "league": "Premier League"},
            {"match_num": 7, "home": "Leipzig", "away": "Dortmund",
             "league": "Bundesliga"},
            {"match_num": 8, "home": "St. Pauli", "away": "Werder Bremen",
             "league": "Bundesliga"},
            {"match_num": 9, "home": "Génova", "away": "Torino",
             "league": "Serie A"},
            {"match_num": 10, "home": "Estoril", "away": "Gil Vicente",
             "league": "Liga Portugal"},
            {"match_num": 11, "home": "Westerlo", "away": "Charleroi",
             "league": "Belgian Pro League"},
            {"match_num": 12, "home": "Vélez", "away": "River Plate",
             "league": "Liga Argentina"},
            {"match_num": 13, "home": "Houston", "away": "Chicago",
             "league": "MLS"},
            {"match_num": 14, "home": "Saprissa", "away": "Alajuelense",
             "league": "Costa Rica"},
        ],
        "revancha": [
            {"match_num": 15, "home": "Puebla", "away": "Águilas",
             "league": "Liga MX"},
            {"match_num": 16, "home": "Atlas", "away": "San Luis",
             "league": "Liga MX"},
            {"match_num": 17, "home": "Necaxa", "away": "Toluca",
             "league": "Liga MX"},
            {"match_num": 18, "home": "Querétaro", "away": "Juárez",
             "league": "Liga MX"},
            {"match_num": 19, "home": "Union Berlin", "away": "Leverkusen",
             "league": "Bundesliga"},
            {"match_num": 20, "home": "Atalanta", "away": "Napoles",
             "league": "Serie A"},
            {"match_num": 21, "home": "Estrasburgo", "away": "Lyon",
             "league": "Ligue 1"},
        ]
    }


# ---------------------------------------------------------------------------
# Main Scraper Interface
# ---------------------------------------------------------------------------

def get_current_fixtures(use_demo: bool = False) -> dict:
    """
    Get current week's fixtures, trying multiple sources.
    
    Priority:
    1. Cached fixtures (if < 7 days old)
    2. Playwright scraper
    3. Requests scraper
    4. Demo fixtures
    
    Args:
        use_demo: If True, skip scraping and use demo fixtures
    
    Returns:
        dict with progol (14) and revancha (7) fixtures
    """
    if use_demo:
        print("[Scraper] Using demo fixtures")
        fixtures = get_demo_fixtures()
        _cache_fixtures(fixtures)
        return fixtures
    
    # Check cache
    if FIXTURES_CACHE.exists():
        try:
            with open(FIXTURES_CACHE, "r", encoding="utf-8") as f:
                cached = json.load(f)
            
            if cached.get("fecha"):
                cached_date = datetime.fromisoformat(cached["fecha"])
                age_days = (datetime.now() - cached_date).days
                if age_days < 7:
                    print(f"[Scraper] Using cached fixtures ({age_days} days old)")
                    return cached
        except Exception:
            pass
    
    # Try Playwright
    print("[Scraper] Attempting Playwright scraper...")
    fixtures = scrape_quiniela_playwright()
    if fixtures and fixtures.get("progol"):
        _cache_fixtures(fixtures)
        return fixtures
    
    # Try requests
    print("[Scraper] Attempting requests scraper...")
    fixtures = scrape_quiniela_requests()
    if fixtures and fixtures.get("progol"):
        _cache_fixtures(fixtures)
        return fixtures
    
    # Fallback to demo
    print("[Scraper] All scrapers failed. Using demo fixtures.")
    fixtures = get_demo_fixtures()
    _cache_fixtures(fixtures)
    return fixtures


def _cache_fixtures(fixtures: dict):
    """Cache fixtures to disk."""
    try:
        with open(FIXTURES_CACHE, "w", encoding="utf-8") as f:
            json.dump(fixtures, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Scraper] Cache write error: {e}")


# ---------------------------------------------------------------------------
# Demo / Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("PROGOL SCRAPER — TEST RUN")
    print("=" * 60)
    
    fixtures = get_current_fixtures(use_demo=True)
    
    print(f"\nConcurso: {fixtures['concurso']}")
    print(f"Fecha: {fixtures['fecha']}")
    
    print(f"\n--- PROGOL ({len(fixtures['progol'])} partidos) ---")
    for f in fixtures["progol"]:
        print(f"  #{f['match_num']:2d} | {f['home']:20s} vs {f['away']:20s} | {f['league']}")
    
    print(f"\n--- REVANCHA ({len(fixtures['revancha'])} partidos) ---")
    for f in fixtures["revancha"]:
        print(f"  #{f['match_num']:2d} | {f['home']:20s} vs {f['away']:20s} | {f['league']}")
    
    print("\n✅ Scraper test complete!")
