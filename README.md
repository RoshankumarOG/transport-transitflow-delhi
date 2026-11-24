# Delhi TransitFlow — Multi-Modal Bus + Metro Router 🚌🚇

An end-to-end intelligent transit routing and ETA prediction system for Delhi, featuring **ML-powered Bus ETA prediction**, **Delhi Metro network integration**, **multi-modal (Bus + Metro) routing**, **fare calculation & cost optimization**, and an **interactive Leaflet.js map**.

![Architecture](https://img.shields.io/badge/ML-LightGBM-blue) ![API](https://img.shields.io/badge/API-FastAPI-green) ![Transit](https://img.shields.io/badge/Network-Bus%20%2B%20Metro-purple) ![Frontend](https://img.shields.io/badge/Frontend-Leaflet.js-orange)

---

## 🌟 Key Features

1. **Multi-Modal Journey Planning (`mixed`)**:
   - Computes pure **Bus-Only**, pure **Metro-Only**, and smart **Bus + Metro** mixed journeys.
   - Models inter-modal transfers (walk connections < 500m between bus stops and metro stations, with headway penalties).

2. **Smart Optimization Modes**:
   - ⚡ **Fastest**: Minimizes end-to-end journey time (takes traffic and metro speed into account).
   - 💰 **Cheapest**: Highlights the most economical transit option (bus fares starting at ₹5 vs metro slabs).
   - 🔄 **Fewest Transfers**: Heavy penalty on mode switches and line changes for the smoothest commute.

3. **Fare Engine**:
   - **Delhi Metro**: Official distance slab fares (₹10, ₹20, ₹30, ₹40, ₹50, ₹60).
   - **DTC / Cluster Buses**: Non-AC (₹5, ₹10, ₹15) and AC (₹10, ₹15, ₹25) distance-tier fares.

4. **ML-Powered Bus ETAs**:
   - LightGBM regressor predicting segment-level bus travel times based on time-of-day, day-of-week, historical speed ratios, and live GPS feeds.

5. **Live Congestion & Transit Overlay**:
   - Interactive Leaflet map displaying live traffic congestion levels on bus routes (green/orange/red) alongside colored Delhi Metro lines (Yellow, Violet, Blue, Magenta).

---

## 🏛️ Architecture

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
                                               ┌────────────┴────────────┐
                                               ▼                         ▼
                                         index.html              /plan-all-options
                                      (Interactive Map)           /plan-journey
```

---

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/RoshankumarOG/transport-transitflow-delhi.git
cd transport-transitflow-delhi

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch FastAPI backend (automatically initializes GTFS + sample data)
uvicorn app:app --reload

# 4. Open index.html in your browser
# Click any origin and destination in Delhi, then click "Compare All Options"!
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/plan-journey` | Plan optimal route with given mode (`bus`, `metro`, `mixed`) and optimization (`fastest`, `cheapest`, `least_transfers`) |
| `POST` | `/plan-all-options` | Returns Bus-only, Metro-only, and Mixed route options simultaneously with ETAs, fares, and badges (`⚡ Fastest`, `💰 Cheapest`) |
| `POST` | `/ingest-live-speed` | Ingest real-time segment speeds from GPS or GTFS-RT feed |
| `GET` | `/segments-status` | Get bus segment speeds and congestion levels (`free_flow`, `moderate`, `heavy`) |
| `GET` | `/metro-lines` | Returns Delhi Metro lines with station coordinates and line colors |
| `GET` | `/metro-stations` | Returns all metro stations with lines and positions |

### Example Request (`POST /plan-all-options`)

```bash
curl -X POST http://127.0.0.1:8000/plan-all-options \
  -H "Content-Type: application/json" \
  -d '{
    "origin_lat": 28.6675,
    "origin_lng": 77.2282,
    "dest_lat": 28.5491,
    "dest_lng": 77.2533,
    "departure_time": "2026-11-24T09:15:00"
  }'
```

---

## 📊 Benchmarks & Validation

Run the benchmark script:

```bash
python evaluate.py
```

### Corridor Comparison (Morning Peak - 9 AM)

| Corridor | Bus ETA | Metro ETA | Mixed ETA | Bus Fare | Metro Fare |
|---|---|---|---|---|---|
| Kashmiri Gate → Nehru Place | 46.5 min | 32.0 min | **30.5 min** ⚡ | ₹15 (Non-AC) 💰 | ₹40 |
| Connaught Place → Hauz Khas | 37.2 min | **18.0 min** ⚡ | 18.0 min | ₹10 (Non-AC) 💰 | ₹30 |
| ITO → Nehru Place | 28.4 min | 24.0 min | **22.0 min** ⚡ | ₹10 (Non-AC) 💰 | ₹30 |

---

## 📁 Project Structure

```text
delhi-bus-router/
├── requirements.txt      # Python dependencies (FastAPI, LightGBM, NetworkX, etc.)
├── data_pipeline.py      # GTFS processing, GPS snapping, Metro network & fare rules
├── model.py              # Historical baseline + LightGBM travel-time model
├── router.py             # Multi-modal transit graph, transfer edges, Dijkstra solver
├── app.py                # FastAPI server (/plan-journey, /plan-all-options, /metro-lines)
├── evaluate.py           # Model MAE/MAPE validation & multi-modal comparisons
├── index.html            # Interactive Leaflet.js demo UI (Bus + Metro overlay)
└── README.md             # Project documentation
```

## 📜 License

MIT
