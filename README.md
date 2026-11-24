# Delhi TransitFlow — Intelligent Bus, Metro & Multi-Modal Router 🚌🚇

An end-to-end intelligent transit routing and ETA prediction system for Delhi, featuring **ML-powered Bus ETA prediction**, **Delhi Metro network integration**, **mode-specific pathfinding (Metro-Only, Bus-Only, Bus + Metro)**, **cost vs. speed optimization (Fastest vs. Cheapest)**, and an **interactive Leaflet.js map**.

![Architecture](https://img.shields.io/badge/ML-LightGBM-blue) ![API](https://img.shields.io/badge/API-FastAPI-green) ![Transit](https://img.shields.io/badge/Network-Bus%20%2B%20Metro-purple) ![Frontend](https://img.shields.io/badge/Frontend-Leaflet.js-orange)

---

## 🌟 Key Capabilities

### 1. 🔀 Three Distinct Commute Modes
Commuters can choose between three transit modes:
- 🚇 **Metro Only**: Strict Delhi Metro rail routes with in-station line interchanges.
- 🚌 **Bus Only**: Direct DTC / cluster bus routes accounting for road congestion and boarding delays.
- 🚌+🚇 **Bus + Metro (Mixed)**: Inter-modal journeys using pedestrian transfer links between bus stops and metro stations within 500m.

### 2. ⚡ vs 💰 Fastest vs. Cheapest Optimization
After selecting any mode, users can toggle between:
- ⚡ **Fastest Way**: Minimizes total travel duration by leveraging high-speed metro lines and low-traffic bus corridors.
- 💰 **Cheapest Way**: Minimizes commute expense via cost-weighted Dijkstra pathfinding (accounting for distance tiers and boarding fees).
- 🔄 **Fewest Transfers**: Minimizes intermediate line changes and mode switches.

### 3. 💵 Official Delhi Fare Calculation
- **Delhi Metro**: Real distance-slab fares (₹10 for $\le 2\text{ km}$, ₹20 for $\le 5\text{ km}$, up to ₹60).
- **DTC / Cluster Buses**: Non-AC (₹5, ₹10, ₹15) and AC (₹10, ₹15, ₹25) distance-tier slabs.

### 4. 🤖 ML Travel-Time & Live Traffic Engine
- **LightGBM Model**: Predicts segment-level bus travel times trained on historical time-of-day, day-of-week, and speed ratios.
- **Dynamic Speed Ingestion**: Live GPS feed updates `/ingest-live-speed` dynamically adjusting bus ETAs and route recommendations.

---

## 🏛️ System Architecture

```
GTFS Schedules + GPS Traces ──► data_pipeline.py ──► Bus Segments + Metro Lines
                                                            │
                                                            ▼
                                                       model.py (LightGBM Bus ETA)
                                                            │
                                                            ▼
                                                       router.py (Multi-Modal DiGraph + Dijkstra)
                                                            │
                                                            ▼
                                                       app.py (FastAPI Backend)
                                                            │
                                  ┌─────────────────────────┼─────────────────────────┐
                                  ▼                         ▼                         ▼
                            index.html           /plan-mode-comparison         /plan-all-options
                         (Interactive Map)        (Fastest vs Cheapest)         (Bus vs Metro vs Mixed)
```

---

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/RoshankumarOG/transport-transitflow-delhi.git
cd transport-transitflow-delhi

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch FastAPI backend (auto-generates GTFS & synthetic GPS baseline on first start)
uvicorn app:app --reload

# 4. Open index.html in any browser
# Pick a mode (Metro, Bus, or Mixed), choose Fastest or Cheapest, and explore routes!
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/plan-mode-comparison` | Returns both **Fastest** and **Cheapest** route options for the chosen mode (`metro`, `bus`, `mixed`) |
| `POST` | `/plan-journey` | Plan an optimal route for a specific mode and optimization goal (`fastest`, `cheapest`, `least_transfers`) |
| `POST` | `/plan-all-options` | Simultaneously computes and compares Bus-Only, Metro-Only, and Mixed itineraries |
| `POST` | `/ingest-live-speed` | Ingest real-time segment speeds from GPS or GTFS-RT feed |
| `GET` | `/segments-status` | Get bus segment speeds and congestion levels (`free_flow`, `moderate`, `heavy`) |
| `GET` | `/metro-lines` | Returns Delhi Metro lines with station coordinates and line colors |
| `GET` | `/metro-stations` | Returns all metro stations with lines and positions |

### Example 1: Compare Fastest vs. Cheapest for a Selected Mode

```bash
curl -X POST http://127.0.0.1:8000/plan-mode-comparison \
  -H "Content-Type: application/json" \
  -d '{
    "origin_lat": 28.6675,
    "origin_lng": 77.2282,
    "dest_lat": 28.5491,
    "dest_lng": 77.2533,
    "departure_time": "2026-11-24T09:15:00",
    "mode": "mixed"
  }'
```

### Example 2: Plan Single Route with Specific Optimization

```bash
curl -X POST http://127.0.0.1:8000/plan-journey \
  -H "Content-Type: application/json" \
  -d '{
    "origin_lat": 28.6315,
    "origin_lng": 77.2167,
    "dest_lat": 28.5494,
    "dest_lng": 77.2001,
    "departure_time": "2026-11-24T18:00:00",
    "mode": "metro",
    "optimize": "fastest"
  }'
```

---

## 📊 Benchmarks & Multi-Modal Comparison

Run the validation suite:

```bash
python evaluate.py
```

### Morning Peak (9 AM) Corridor Benchmarks

| Corridor | 🚌 Bus Only | 🚇 Metro Only | 🚌+🚇 Best Mixed | 💰 Cheapest Fare | ⚡ Fastest Time |
|---|---|---|---|---|---|
| **Kashmiri Gate ➔ Nehru Place** | 46.5 min (₹15) | 32.0 min (₹40) | **30.5 min** (₹40) | ₹15 (Bus) | 30.5 min (Mixed) |
| **Connaught Place ➔ Hauz Khas** | 37.2 min (₹10) | **18.0 min** (₹30) | 18.0 min (₹30) | ₹10 (Bus) | 18.0 min (Metro) |
| **ITO ➔ Nehru Place** | 28.4 min (₹10) | 24.0 min (₹30) | **22.0 min** (₹30) | ₹10 (Bus) | 22.0 min (Mixed) |

---

## 🗺️ Interactive Demo UI

Open `index.html` locally:
- **1-Click Presets**: Test key corridors (*Kashmiri Gate ➔ Nehru Place*, *Connaught Place ➔ Hauz Khas*, *ITO ➔ Nehru Place*) instantly.
- **Segmented Mode Switcher**: Select between Metro Only, Bus Only, or Bus + Metro.
- **Fastest vs. Cheapest Cards**: Side-by-side cards highlighting commute time and fare, with one-click route switching.
- **Live Overlays**: Colored Delhi Metro network (Yellow, Violet, Blue, Magenta) and congestion color codes on bus corridors.

---

## 📁 Repository Structure

```text
delhi-bus-router/
├── requirements.txt      # Python dependencies (FastAPI, LightGBM, NetworkX, etc.)
├── data_pipeline.py      # GTFS processing, GPS snapping, Metro network & fare tables
├── model.py              # Historical baseline + LightGBM travel-time model
├── router.py             # Multi-modal transit graph, transfer edges, Dijkstra solver
├── app.py                # FastAPI server (/plan-journey, /plan-mode-comparison, /metro-lines)
├── evaluate.py           # Model MAE/MAPE validation & multi-modal comparisons
├── index.html            # Interactive Leaflet.js demo UI (Bus + Metro overlay)
└── README.md             # Project documentation
```

## 📜 License

MIT
