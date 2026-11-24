import os
from datetime import datetime
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from data_pipeline import build_segments_from_gtfs, extract_observed_segment_times, build_metro_network
from model import TrafficETAModel
from router import TransitRouter

app = FastAPI(title="Delhi Multi-Modal Transit API (Bus + Metro)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global state
eta_model = TrafficETAModel()
transit_router = None
segments_df = None
metro_stations_df = None
metro_segments_df = None
live_speeds_cache = {}

def generate_sample_delhi_data():
    """Creates realistic Delhi corridor GTFS + GPS logs if data/gtfs is missing."""
    os.makedirs("data/gtfs", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    stops = pd.DataFrame([
        {"stop_id": 1, "stop_name": "Kashmiri Gate ISBT", "stop_lat": 28.6675, "stop_lon": 77.2282},
        {"stop_id": 2, "stop_name": "Red Fort", "stop_lat": 28.6562, "stop_lon": 77.2410},
        {"stop_id": 3, "stop_name": "ITO", "stop_lat": 28.6289, "stop_lon": 77.2405},
        {"stop_id": 4, "stop_name": "Connaught Place", "stop_lat": 28.6315, "stop_lon": 77.2167},
        {"stop_id": 5, "stop_name": "AIIMS", "stop_lat": 28.5672, "stop_lon": 77.2100},
        {"stop_id": 6, "stop_name": "Hauz Khas", "stop_lat": 28.5494, "stop_lon": 77.2001},
        {"stop_id": 7, "stop_name": "Nehru Place", "stop_lat": 28.5491, "stop_lon": 77.2533},
    ])
    routes = pd.DataFrame([
        {"route_id": "R419", "route_short_name": "419"},
        {"route_id": "R502", "route_short_name": "502"},
        {"route_id": "R764", "route_short_name": "764"},
    ])
    trips = pd.DataFrame([
        {"trip_id": "T1", "route_id": "R419"},
        {"trip_id": "T2", "route_id": "R502"},
        {"trip_id": "T3", "route_id": "R764"},
    ])
    stop_times = pd.DataFrame([
        {"trip_id": "T1", "stop_id": 1, "stop_sequence": 1},
        {"trip_id": "T1", "stop_id": 2, "stop_sequence": 2},
        {"trip_id": "T1", "stop_id": 3, "stop_sequence": 3},
        {"trip_id": "T1", "stop_id": 7, "stop_sequence": 4},
        {"trip_id": "T2", "stop_id": 4, "stop_sequence": 1},
        {"trip_id": "T2", "stop_id": 5, "stop_sequence": 2},
        {"trip_id": "T2", "stop_id": 6, "stop_sequence": 3},
        {"trip_id": "T3", "stop_id": 3, "stop_sequence": 1},
        {"trip_id": "T3", "stop_id": 4, "stop_sequence": 2},
        {"trip_id": "T3", "stop_id": 6, "stop_sequence": 3},
        {"trip_id": "T3", "stop_id": 7, "stop_sequence": 4},
    ])
    stops.to_csv("data/gtfs/stops.txt", index=False)
    routes.to_csv("data/gtfs/routes.txt", index=False)
    trips.to_csv("data/gtfs/trips.txt", index=False)
    stop_times.to_csv("data/gtfs/stop_times.txt", index=False)

    rng = np.random.default_rng(42)
    rows = []
    for hour in range(6, 23):
        rush_factor = 1.65 if hour in [8, 9, 10, 17, 18, 19, 20] else 1.0
        for dow in range(7):
            for route_id, seq in [("R419", [1, 2, 3, 7]), ("R502", [4, 5, 6]), ("R764", [3, 4, 6, 7])]:
                t = pd.Timestamp(f"2026-11-23 {hour:02d}:00:00") + pd.Timedelta(days=dow)
                for sid in seq:
                    s_row = stops[stops["stop_id"] == sid].iloc[0]
                    rows.append({
                        "vehicle_id": f"DL1PC_{route_id}_{hour}_{dow}",
                        "route_id": route_id,
                        "timestamp": t,
                        "lat": s_row["stop_lat"],
                        "lon": s_row["stop_lon"],
                        "speed_kmph": 22.0 / rush_factor
                    })
                    t += pd.Timedelta(minutes=float(rng.uniform(7, 12) * rush_factor))
    pd.DataFrame(rows).to_csv("data/sample_gps.csv", index=False)

@app.on_event("startup")
def startup_event():
    global transit_router, segments_df, metro_stations_df, metro_segments_df

    if not os.path.exists("data/gtfs/stops.txt"):
        generate_sample_delhi_data()

    stops_df, segments_df = build_segments_from_gtfs("data/gtfs")
    metro_stations_df, metro_segments_df, metro_interchanges = build_metro_network()

    if os.path.exists("artifacts/eta_model.joblib"):
        eta_model.load("artifacts/eta_model.joblib")
    else:
        gps_df = pd.read_csv("data/sample_gps.csv")
        obs_df = extract_observed_segment_times(gps_df, stops_df, segments_df)
        eta_model.train(obs_df)
        eta_model.save("artifacts/eta_model.joblib")

    transit_router = TransitRouter(stops_df, segments_df, metro_stations_df, metro_segments_df, metro_interchanges)

class JourneyRequest(BaseModel):
    origin_lat: float
    origin_lng: float
    dest_lat: float
    dest_lng: float
    departure_time: str = "2026-11-24T09:15:00"
    mode: str = "mixed"           # bus | metro | mixed
    optimize: str = "fastest"     # fastest | cheapest | least_transfers

class LiveGPSPing(BaseModel):
    segment_id: str
    speed_kmph: float

@app.post("/ingest-live-speed")
def ingest_live_speed(pings: list[LiveGPSPing]):
    for p in pings:
        live_speeds_cache[p.segment_id] = p.speed_kmph
    return {"updated_segments": len(pings)}

@app.post("/plan-journey")
def plan_journey(req: JourneyRequest):
    dt = datetime.fromisoformat(req.departure_time)
    scored_segments = eta_model.predict_segments(
        segments_df, hour=dt.hour, day_of_week=dt.weekday(), live_speeds_map=live_speeds_cache
    )
    graph = transit_router.build_graph(scored_segments, mode=req.mode)
    result = transit_router.plan_journey(
        graph, req.origin_lat, req.origin_lng, req.dest_lat, req.dest_lng,
        mode=req.mode, optimize=req.optimize
    )
    return result

@app.post("/plan-all-options")
def plan_all_options(req: JourneyRequest):
    """Returns bus-only, metro-only, and mixed route options with fares."""
    dt = datetime.fromisoformat(req.departure_time)
    scored_segments = eta_model.predict_segments(
        segments_df, hour=dt.hour, day_of_week=dt.weekday(), live_speeds_map=live_speeds_cache
    )
    options = transit_router.plan_all_options(
        scored_segments, req.origin_lat, req.origin_lng, req.dest_lat, req.dest_lng
    )
    return {"options": options}

@app.get("/segments-status")
def segments_status(hour: int = 9, day_of_week: int = 1):
    scored = eta_model.predict_segments(segments_df, hour, day_of_week, live_speeds_cache)
    return scored[[
        "segment_id", "route_short_name", "stop_lat_from", "stop_lon_from",
        "stop_lat_to", "stop_lon_to", "pred_speed_kmph", "congestion"
    ]].to_dict(orient="records")

@app.get("/metro-lines")
def metro_lines():
    """Returns metro line geometries for map overlay."""
    lines = {}
    for _, row in metro_segments_df.iterrows():
        line = row["route_short_name"]
        if line not in lines:
            lines[line] = {"name": line, "segments": [], "color": {
                "Metro Yellow": "#FFD700", "Metro Violet": "#8B008B",
                "Metro Blue": "#1E90FF", "Metro Magenta": "#FF00FF"
            }.get(line, "#888888")}
        lines[line]["segments"].append({
            "from": [row["stop_lat_from"], row["stop_lon_from"]],
            "to": [row["stop_lat_to"], row["stop_lon_to"]],
            "station_from": row["stop_name_from"],
            "station_to": row["stop_name_to"]
        })
    return list(lines.values())

@app.get("/metro-stations")
def metro_stations():
    """Returns all metro stations for map markers."""
    return metro_stations_df[["station_id", "station_name", "stop_lat", "stop_lon", "line"]].to_dict(orient="records")
