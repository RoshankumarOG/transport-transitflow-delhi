# Delhi Bus ETA & Transit Router 🚌

An end-to-end intelligent bus transit routing system for Delhi, featuring **ML-powered ETA prediction**, **real-time congestion awareness**, and an **interactive map UI**.

![Architecture](https://img.shields.io/badge/ML-LightGBM-blue) ![API](https://img.shields.io/badge/API-FastAPI-green) ![Frontend](https://img.shields.io/badge/Frontend-Leaflet.js-orange)

## Architecture

```
GPS Traces + GTFS ──► data_pipeline.py ──► Segment Travel Times
                                               │
                                               ▼
                                          model.py (LightGBM ETA)
                                               │
                                               ▼
                                          router.py (Dijkstra + Transfers)
                                               │
                                               ▼
                                          app.py (FastAPI Backend)
                                               │
                                    ┌──────────┴──────────┐
                                    ▼                     ▼
                              index.html            /plan-journey API
                           (Leaflet.js Map)
```

## Features

- **GTFS Data Pipeline** — Parses standard GTFS feeds into stop-to-stop segments, snaps raw GPS pings to compute observed travel times
- **LightGBM ETA Model** — Combines historical hour/day baselines with recent live GPS speeds for accurate bus travel time prediction
- **Multi-Modal Transit Router** — NetworkX DiGraph with transit edges, same-stop route transfers, and walking transfers (< 400m) using Time-Dependent Dijkstra
- **Real-Time Congestion** — Live speed ingestion via `/ingest-live-speed` endpoint dynamically adjusts ETAs
- **Interactive Map** — Dark-themed Leaflet.js UI with click-to-route, congestion color overlays, and step-by-step journey legs

## Quick Start

```bash
# Clone the repo
git clone https://github.com/<your-username>/delhi-bus-router.git
cd delhi-bus-router

# Install dependencies
pip install -r requirements.txt

# Start the API server (auto-generates sample Delhi data on first run)
uvicorn app:app --reload

# Open index.html in your browser and click two points on the Delhi map!
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/plan-journey` | Plan an optimal bus route between two coordinates |
| `POST` | `/ingest-live-speed` | Update real-time segment speeds from GPS/GTFS-RT |
| `GET` | `/segments-status` | Get all segments with congestion levels for map overlay |

### Example: Plan a Journey

```bash
curl -X POST http://127.0.0.1:8000/plan-journey \
  -H "Content-Type: application/json" \
  -d '{
    "origin_lat": 28.6675,
    "origin_lng": 77.2282,
    "dest_lat": 28.5491,
    "dest_lng": 77.2533,
    "departure_time": "2026-11-24T09:15:00"
  }'
```

## Validation

Run the benchmark script to see MAE/MAPE on held-out test data and corridor comparisons vs Google Maps:

```bash
python evaluate.py
```

## Using Real Delhi Data

1. Drop official **Delhi OTD GTFS** files (`stops.txt`, `routes.txt`, `trips.txt`, `stop_times.txt`) into `data/gtfs/`
2. Format historical GPS logs as CSV: `vehicle_id, route_id, timestamp, lat, lon, speed_kmph`
3. Delete `artifacts/eta_model.joblib` so the model retrains on next server start
4. Connect your live GTFS-RT feed to `POST /ingest-live-speed` every 30–60 seconds

## Project Structure

```
delhi-bus-router/
├── requirements.txt      # Python dependencies
├── data_pipeline.py      # GTFS parsing, GPS snapping, segment travel times
├── model.py              # Historical baseline + LightGBM ETA predictor
├── router.py             # Transit graph, transfer edges, Dijkstra planner
├── app.py                # FastAPI backend (/plan-journey)
├── evaluate.py           # Validation: MAE, MAPE & Google Maps comparison
└── index.html            # Interactive Leaflet.js demo map
```

## Tech Stack

- **ML**: LightGBM, scikit-learn, pandas, NumPy
- **Routing**: NetworkX (Dijkstra), SciPy (cKDTree spatial indexing)
- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: Leaflet.js, CARTO dark basemap
- **Data**: GTFS standard, GPS traces with Haversine distance

## License

MIT
