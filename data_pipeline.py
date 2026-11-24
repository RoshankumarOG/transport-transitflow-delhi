import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def build_segments_from_gtfs(gtfs_dir="data/gtfs"):
    """Parses GTFS into unique stop-to-stop segments per route."""
    stops = pd.read_csv(f"{gtfs_dir}/stops.txt")
    routes = pd.read_csv(f"{gtfs_dir}/routes.txt")
    trips = pd.read_csv(f"{gtfs_dir}/trips.txt")
    stop_times = pd.read_csv(f"{gtfs_dir}/stop_times.txt")

    # Pick one canonical trip per route to extract stop sequence
    canonical_trips = trips.drop_duplicates("route_id")[["trip_id", "route_id"]]
    st = stop_times.merge(canonical_trips, on="trip_id").sort_values(["route_id", "stop_sequence"])
    st = st.merge(routes[["route_id", "route_short_name"]], on="route_id")

    st["next_stop_id"] = st.groupby("route_id")["stop_id"].shift(-1)
    segments = st.dropna(subset=["next_stop_id"]).copy()
    segments["next_stop_id"] = segments["next_stop_id"].astype(stops["stop_id"].dtype)

    # Merge coordinates for start and end stops
    segments = segments.merge(stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]], on="stop_id")
    segments = segments.merge(
        stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]],
        left_on="next_stop_id", right_on="stop_id", suffixes=("_from", "_to")
    )

    segments["distance_m"] = haversine_m(
        segments["stop_lat_from"], segments["stop_lon_from"],
        segments["stop_lat_to"], segments["stop_lon_to"]
    )
    segments["segment_id"] = (
        segments["route_id"].astype(str) + "_" +
        segments["stop_id_from"].astype(str) + "_" +
        segments["stop_id_to"].astype(str)
    )
    return stops, segments

def extract_observed_segment_times(gps_df, stops_df, segments_df):
    """
    Filters stationary pings, snaps GPS traces to nearest stops within 60m,
    and calculates observed segment travel times.
    """
    gps_df = gps_df.sort_values(["vehicle_id", "timestamp"]).copy()
    gps_df["timestamp"] = pd.to_datetime(gps_df["timestamp"])

    # Filter stationary depot noise (speed < 1 km/h for extended stops)
    if "speed_kmph" in gps_df.columns:
        gps_df = gps_df[gps_df["speed_kmph"] > 1.0].copy()

    # Spatial snap to nearest stop using KDTree
    stop_coords = np.radians(stops_df[["stop_lat", "stop_lon"]].values)
    tree = cKDTree(stop_coords)
    dists, idxs = tree.query(np.radians(gps_df[["lat", "lon"]].values), k=1)

    gps_df["dist_to_stop_m"] = dists * 6371000.0
    gps_df["snapped_stop_id"] = stops_df.iloc[idxs]["stop_id"].values

    # Keep pings within 60 meters of a stop
    arrivals = gps_df[gps_df["dist_to_stop_m"] <= 60].copy()
    # Deduplicate consecutive pings at the same stop
    arrivals = arrivals[arrivals["snapped_stop_id"] != arrivals.groupby("vehicle_id")["snapped_stop_id"].shift(1)]

    arrivals["next_stop_id"] = arrivals.groupby("vehicle_id")["snapped_stop_id"].shift(-1)
    arrivals["next_time"] = arrivals.groupby("vehicle_id")["timestamp"].shift(-1)
    arrivals["travel_time_sec"] = (arrivals["next_time"] - arrivals["timestamp"]).dt.total_seconds()

    # Join with valid route segments
    obs = arrivals.merge(
        segments_df[["segment_id", "route_id", "stop_id_from", "stop_id_to", "distance_m"]],
        left_on=["route_id", "snapped_stop_id", "next_stop_id"],
        right_on=["route_id", "stop_id_from", "stop_id_to"]
    )

    # Filter unrealistic outliers (speeds < 2 km/h or > 70 km/h)
    obs["speed_kmph"] = (obs["distance_m"] / 1000.0) / (obs["travel_time_sec"] / 3600.0)
    obs = obs[(obs["speed_kmph"] >= 2.0) & (obs["speed_kmph"] <= 70.0)].copy()

    obs["hour"] = obs["timestamp"].dt.hour
    obs["day_of_week"] = obs["timestamp"].dt.dayofweek
    return obs
