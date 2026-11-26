import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from data_pipeline import build_segments_from_gtfs, extract_observed_segment_times, build_metro_network
from model import TrafficETAModel
from router import TransitRouter

def run_benchmark():
    stops_df, segments_df = build_segments_from_gtfs("data/gtfs")
    metro_stations_df, metro_segments_df, metro_interchanges = build_metro_network()
    gps_df = pd.read_csv("data/sample_gps.csv")
    obs_df = extract_observed_segment_times(gps_df, stops_df, segments_df)

    # 80/20 Train-Test Split
    train_df, test_df = train_test_split(obs_df, test_size=0.20, random_state=42)

    model = TrafficETAModel()
    model.train(train_df)

    # Evaluate on held-out test set
    preds = []
    for _, row in test_df.iterrows():
        sub = segments_df[segments_df["segment_id"] == row["segment_id"]]
        out = model.predict_segments(sub, int(row["hour"]), int(row["day_of_week"]))
        preds.append(out["pred_travel_time_sec"].values[0])

    test_df["pred_sec"] = preds
    mae_min = mean_absolute_error(test_df["travel_time_sec"] / 60.0, test_df["pred_sec"] / 60.0)
    mape_pct = mean_absolute_percentage_error(test_df["travel_time_sec"], test_df["pred_sec"]) * 100.0

    print("=== HELD-OUT SYNTHETIC BUS TRIPS BENCHMARK ===")
    print(f"Test Samples : {len(test_df)} segment traversals")
    print(f"MAE          : {mae_min:.2f} minutes")
    print(f"MAPE         : {mape_pct:.2f}%\n")

    # Multi-Modal Corridor Comparison Table (Bus vs Metro vs Mixed vs Google Maps)
    corridors = pd.DataFrame([
        {
            "Corridor": "Kashmiri Gate -> Nehru Place (9 AM)",
            "Bus_ETA_Min": 46.5,
            "Metro_ETA_Min": 32.0,
            "Mixed_ETA_Min": 30.5,
            "Bus_Fare_₹": "15 (Non-AC) / 25 (AC)",
            "Metro_Fare_₹": "40",
            "Google_Maps_Car_Min": 34.0
        },
        {
            "Corridor": "Connaught Place -> Hauz Khas (6 PM)",
            "Bus_ETA_Min": 37.2,
            "Metro_ETA_Min": 18.0,
            "Mixed_ETA_Min": 18.0,
            "Bus_Fare_₹": "10 (Non-AC) / 15 (AC)",
            "Metro_Fare_₹": "30",
            "Google_Maps_Car_Min": 25.0
        },
        {
            "Corridor": "ITO -> Nehru Place (2 PM)",
            "Bus_ETA_Min": 28.4,
            "Metro_ETA_Min": 24.0,
            "Mixed_ETA_Min": 22.0,
            "Bus_Fare_₹": "10 (Non-AC) / 15 (AC)",
            "Metro_Fare_₹": "30",
            "Google_Maps_Car_Min": 22.0
        },
    ])
    print("=== ILLUSTRATIVE CORRIDOR COMPARISON (NOT LIVE MEASUREMENTS) ===")
    print(corridors.to_string(index=False))

    # Multi-Modal Router Verification
    router = TransitRouter(stops_df, segments_df, metro_stations_df, metro_segments_df, metro_interchanges)
    scored = model.predict_segments(segments_df, hour=9, day_of_week=1)
    test_options = router.plan_all_options(
        scored, orig_lat=28.6675, orig_lng=77.2282, dest_lat=28.5491, dest_lng=77.2533
    )
    print(f"\n=== MULTI-MODAL ROUTING TEST: Kashmiri Gate -> Nehru Place ===")
    for opt in test_options:
        print(f"[{opt['option_label']}] ETA: {opt['total_eta_min']}m | Transfers: {opt['transfers']} | Fare: ₹{opt['fare']['total_fare_min']}–₹{opt['fare']['total_fare_max']} | Tags: {opt.get('tags', [])}")

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    run_benchmark()
