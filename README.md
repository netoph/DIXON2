# ⚽ Progol Dashboard — Dixon-Coles Prediction Engine

<p align="center">
  <strong>Automated predictive dashboard for Progol and Revancha lotteries</strong><br>
  Powered by the Dixon-Coles (1997) statistical model for football match outcomes
</p>

---

## 🎯 What is this?

An end-to-end system that:

1. **Scrapes** weekly Progol quiniela fixtures from Lotería Nacional
2. **Fetches** historical match data from FBref (via soccerdata) + fallback sources
3. **Fits** the Dixon-Coles model with time-decay weighting
4. **Predicts** match outcomes (Home/Draw/Away) with probability distributions
5. **Displays** everything in a premium dark-themed React dashboard

## 🏗️ Architecture

```
┌─────────────────────┐     ┌──────────────────────┐
│   React Frontend    │────▶│   FastAPI Backend     │
│   (Vite + Tailwind) │◀────│   (Port 8000)        │
│   Port 5173         │     │                      │
└─────────────────────┘     │  ┌────────────────┐  │
                            │  │ Dixon-Coles    │  │
                            │  │ Engine         │  │
                            │  └────────────────┘  │
                            │  ┌────────────────┐  │
                            │  │ Data Fetcher   │  │
                            │  │ (FBref/Web)    │  │
                            │  └────────────────┘  │
                            │  ┌────────────────┐  │
                            │  │ Progol Scraper │  │
                            │  │ (Lotería Nal.) │  │
                            │  └────────────────┘  │
                            └──────────────────────┘
```

## 🚀 Quick Start

### Backend

```bash
cd backend
pip install -r requirements.txt
python server.py
# API running on http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Dashboard on http://localhost:5173
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predictions` | GET | Full predictions (Progol 14 + Revancha 7) |
| `/api/status` | GET | Model parameters (rho, xi, convergence) |
| `/api/refresh` | POST | Trigger pipeline re-run |

## 🧮 Dixon-Coles Model

The core prediction engine implements:

- **Poisson goal distribution** for each team
- **Rho correction** for low-scoring outcomes (0-0, 0-1, 1-0, 1-1)
- **Time-decay weighting** (ξ parameter) — recent form matters more
- **Vectorized log-likelihood** for fast optimization (~3s for 1700 matches)
- **Zero-NaN Policy** — strict data quality enforcement

### Output per match

- Home win / Draw / Away win probabilities
- Recommended pick: **L** (Local), **E** (Empate), **V** (Visitante)
- Top 3 most likely scorelines with probabilities

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Prediction Engine** | Python, SciPy, NumPy, Pandas |
| **Backend API** | FastAPI, Uvicorn, APScheduler |
| **Data Sources** | FBref (soccerdata), football-data.org, API-Football |
| **Scraper** | Playwright, BeautifulSoup4, Requests |
| **Frontend** | React 18, Vite, Tailwind CSS |
| **Scheduling** | APScheduler (Monday 09:00 CST) |

## 📁 Project Structure

```
Dixon Coles/
├── backend/
│   ├── dc_prediction_progol.py   # Dixon-Coles engine
│   ├── data_fetcher.py           # Multi-source data pipeline
│   ├── progol_scraper.py         # Lotería Nacional scraper
│   ├── server.py                 # FastAPI server + scheduler
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx               # Main app component
│   │   ├── index.css             # Dark theme + animations
│   │   └── components/
│   │       ├── Header.jsx        # Glassmorphism header
│   │       ├── ModelStatus.jsx   # Model parameters display
│   │       └── PredictionTable.jsx # Match predictions grid
│   ├── index.html
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── package.json
└── README.md
```

## 📄 License

MIT

---

<p align="center">
  Built with 🧠 Dixon-Coles + ⚡ FastAPI + ⚛️ React
</p>
