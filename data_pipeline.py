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

    canonical_trips = trips.drop_duplicates("route_id")[["trip_id", "route_id"]]
    st = stop_times.merge(canonical_trips, on="trip_id").sort_values(["route_id", "stop_sequence"])
    st = st.merge(routes[["route_id", "route_short_name"]], on="route_id")

    st["next_stop_id"] = st.groupby("route_id")["stop_id"].shift(-1)
    segments = st.dropna(subset=["next_stop_id"]).copy()
    segments["next_stop_id"] = segments["next_stop_id"].astype(stops["stop_id"].dtype)

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
    """Snaps GPS traces to nearest stops and calculates observed segment travel times."""
    gps_df = gps_df.sort_values(["vehicle_id", "timestamp"]).copy()
    gps_df["timestamp"] = pd.to_datetime(gps_df["timestamp"])

    if "speed_kmph" in gps_df.columns:
        gps_df = gps_df[gps_df["speed_kmph"] > 1.0].copy()

    stop_coords = np.radians(stops_df[["stop_lat", "stop_lon"]].values)
    tree = cKDTree(stop_coords)
    dists, idxs = tree.query(np.radians(gps_df[["lat", "lon"]].values), k=1)

    gps_df["dist_to_stop_m"] = dists * 6371000.0
    gps_df["snapped_stop_id"] = stops_df.iloc[idxs]["stop_id"].values

    arrivals = gps_df[gps_df["dist_to_stop_m"] <= 60].copy()
    arrivals = arrivals[arrivals["snapped_stop_id"] != arrivals.groupby("vehicle_id")["snapped_stop_id"].shift(1)]

    arrivals["next_stop_id"] = arrivals.groupby("vehicle_id")["snapped_stop_id"].shift(-1)
    arrivals["next_time"] = arrivals.groupby("vehicle_id")["timestamp"].shift(-1)
    arrivals["travel_time_sec"] = (arrivals["next_time"] - arrivals["timestamp"]).dt.total_seconds()

    obs = arrivals.merge(
        segments_df[["segment_id", "route_id", "stop_id_from", "stop_id_to", "distance_m"]],
        left_on=["route_id", "snapped_stop_id", "next_stop_id"],
        right_on=["route_id", "stop_id_from", "stop_id_to"]
    )

    obs["speed_kmph"] = (obs["distance_m"] / 1000.0) / (obs["travel_time_sec"] / 3600.0)
    obs = obs[(obs["speed_kmph"] >= 2.0) & (obs["speed_kmph"] <= 70.0)].copy()

    obs["hour"] = obs["timestamp"].dt.hour
    obs["day_of_week"] = obs["timestamp"].dt.dayofweek
    return obs


# ───────────────────────────────────────────────────────────
#  DELHI METRO NETWORK
# ───────────────────────────────────────────────────────────

def build_metro_network():
    """
    Returns metro_stations (DataFrame) and metro_segments (DataFrame)
    for the Delhi Metro network.
    Metro travel times are fixed (~2 min inter-station + 30s dwell).
    """

    # Key stations across Yellow, Violet, Blue, Magenta lines
    # covering the corridor our bus routes serve
    metro_stations = pd.DataFrame([
        # Yellow Line stations
        {"station_id": "M_Y01", "station_name": "Kashmere Gate",        "stop_lat": 28.6674, "stop_lon": 77.2289, "line": "Yellow"},
        {"station_id": "M_Y02", "station_name": "Chandni Chowk",       "stop_lat": 28.6506, "stop_lon": 77.2303, "line": "Yellow"},
        {"station_id": "M_Y03", "station_name": "Chawri Bazar",        "stop_lat": 28.6451, "stop_lon": 77.2254, "line": "Yellow"},
        {"station_id": "M_Y04", "station_name": "New Delhi",           "stop_lat": 28.6424, "stop_lon": 77.2195, "line": "Yellow"},
        {"station_id": "M_Y05", "station_name": "Rajiv Chowk",         "stop_lat": 28.6328, "stop_lon": 77.2197, "line": "Yellow"},
        {"station_id": "M_Y06", "station_name": "Patel Chowk",         "stop_lat": 28.6250, "stop_lon": 77.2131, "line": "Yellow"},
        {"station_id": "M_Y07", "station_name": "Central Secretariat", "stop_lat": 28.6148, "stop_lon": 77.2114, "line": "Yellow"},
        {"station_id": "M_Y08", "station_name": "Udyog Bhawan",        "stop_lat": 28.6098, "stop_lon": 77.2098, "line": "Yellow"},
        {"station_id": "M_Y09", "station_name": "Lok Kalyan Marg",     "stop_lat": 28.6002, "stop_lon": 77.2051, "line": "Yellow"},
        {"station_id": "M_Y10", "station_name": "Jor Bagh",            "stop_lat": 28.5898, "stop_lon": 77.2032, "line": "Yellow"},
        {"station_id": "M_Y11", "station_name": "INA",                 "stop_lat": 28.5789, "stop_lon": 77.2074, "line": "Yellow"},
        {"station_id": "M_Y12", "station_name": "AIIMS Metro",         "stop_lat": 28.5685, "stop_lon": 77.2079, "line": "Yellow"},
        {"station_id": "M_Y13", "station_name": "Green Park",          "stop_lat": 28.5589, "stop_lon": 77.2067, "line": "Yellow"},
        {"station_id": "M_Y14", "station_name": "Hauz Khas Metro",     "stop_lat": 28.5494, "stop_lon": 77.2065, "line": "Yellow"},

        # Violet Line stations
        {"station_id": "M_V01", "station_name": "Kashmere Gate (V)",    "stop_lat": 28.6672, "stop_lon": 77.2291, "line": "Violet"},
        {"station_id": "M_V02", "station_name": "Lal Quila",           "stop_lat": 28.6558, "stop_lon": 77.2412, "line": "Violet"},
        {"station_id": "M_V03", "station_name": "Jama Masjid",         "stop_lat": 28.6508, "stop_lon": 77.2365, "line": "Violet"},
        {"station_id": "M_V04", "station_name": "Delhi Gate",          "stop_lat": 28.6420, "stop_lon": 77.2412, "line": "Violet"},
        {"station_id": "M_V05", "station_name": "ITO Metro",           "stop_lat": 28.6300, "stop_lon": 77.2405, "line": "Violet"},
        {"station_id": "M_V06", "station_name": "Mandi House",         "stop_lat": 28.6260, "stop_lon": 77.2335, "line": "Violet"},
        {"station_id": "M_V07", "station_name": "Janpath",             "stop_lat": 28.6238, "stop_lon": 77.2185, "line": "Violet"},
        {"station_id": "M_V08", "station_name": "Central Secretariat (V)", "stop_lat": 28.6150, "stop_lon": 77.2116, "line": "Violet"},
        {"station_id": "M_V09", "station_name": "Khan Market",         "stop_lat": 28.6002, "stop_lon": 77.2270, "line": "Violet"},
        {"station_id": "M_V10", "station_name": "JLN Stadium",         "stop_lat": 28.5882, "stop_lon": 77.2370, "line": "Violet"},
        {"station_id": "M_V11", "station_name": "Lajpat Nagar",        "stop_lat": 28.5700, "stop_lon": 77.2432, "line": "Violet"},
        {"station_id": "M_V12", "station_name": "Moolchand",           "stop_lat": 28.5635, "stop_lon": 77.2425, "line": "Violet"},
        {"station_id": "M_V13", "station_name": "Nehru Place Metro",   "stop_lat": 28.5490, "stop_lon": 77.2512, "line": "Violet"},

        # Blue Line stations (key interchange points)
        {"station_id": "M_B01", "station_name": "Rajiv Chowk (B)",     "stop_lat": 28.6330, "stop_lon": 77.2199, "line": "Blue"},
        {"station_id": "M_B02", "station_name": "Barakhamba Road",     "stop_lat": 28.6350, "stop_lon": 77.2270, "line": "Blue"},
        {"station_id": "M_B03", "station_name": "Mandi House (B)",     "stop_lat": 28.6262, "stop_lon": 77.2337, "line": "Blue"},
        {"station_id": "M_B04", "station_name": "Pragati Maidan",      "stop_lat": 28.6218, "stop_lon": 77.2460, "line": "Blue"},
        {"station_id": "M_B05", "station_name": "Indraprastha",        "stop_lat": 28.6180, "stop_lon": 77.2500, "line": "Blue"},

        # Magenta Line (Hauz Khas interchange)
        {"station_id": "M_M01", "station_name": "Hauz Khas (M)",       "stop_lat": 28.5496, "stop_lon": 77.2063, "line": "Magenta"},
        {"station_id": "M_M02", "station_name": "Panchsheel Park",     "stop_lat": 28.5420, "stop_lon": 77.2170, "line": "Magenta"},
        {"station_id": "M_M03", "station_name": "Chirag Delhi",        "stop_lat": 28.5380, "stop_lon": 77.2260, "line": "Magenta"},
        {"station_id": "M_M04", "station_name": "Nehru Enclave",       "stop_lat": 28.5420, "stop_lon": 77.2440, "line": "Magenta"},
    ])

    # Build metro segments (sequential station-to-station within each line)
    METRO_INTERSTATION_SEC = 120   # ~2 minutes between stations
    METRO_DWELL_SEC = 30           # 30s dwell at each station

    segments = []
    for line in metro_stations["line"].unique():
        line_stations = metro_stations[metro_stations["line"] == line].reset_index(drop=True)
        for i in range(len(line_stations) - 1):
            s1 = line_stations.iloc[i]
            s2 = line_stations.iloc[i + 1]
            dist = haversine_m(s1["stop_lat"], s1["stop_lon"], s2["stop_lat"], s2["stop_lon"])
            travel_sec = METRO_INTERSTATION_SEC + METRO_DWELL_SEC
            seg_id = f"METRO_{line}_{s1['station_id']}_{s2['station_id']}"

            # Forward direction
            segments.append({
                "segment_id": seg_id,
                "route_id": f"METRO_{line}",
                "route_short_name": f"Metro {line}",
                "stop_id_from": s1["station_id"],
                "stop_id_to": s2["station_id"],
                "stop_name_from": s1["station_name"],
                "stop_name_to": s2["station_name"],
                "stop_lat_from": s1["stop_lat"],
                "stop_lon_from": s1["stop_lon"],
                "stop_lat_to": s2["stop_lat"],
                "stop_lon_to": s2["stop_lon"],
                "distance_m": dist,
                "travel_time_sec": travel_sec,
                "mode": "metro"
            })

            # Reverse direction
            segments.append({
                "segment_id": f"METRO_{line}_{s2['station_id']}_{s1['station_id']}",
                "route_id": f"METRO_{line}",
                "route_short_name": f"Metro {line}",
                "stop_id_from": s2["station_id"],
                "stop_id_to": s1["station_id"],
                "stop_name_from": s2["station_name"],
                "stop_name_to": s1["station_name"],
                "stop_lat_from": s2["stop_lat"],
                "stop_lon_from": s2["stop_lon"],
                "stop_lat_to": s1["stop_lat"],
                "stop_lon_to": s1["stop_lon"],
                "distance_m": dist,
                "travel_time_sec": travel_sec,
                "mode": "metro"
            })

    metro_segments = pd.DataFrame(segments)

    # Interchange edges (same physical station, different lines)
    # Walking within station ~120 seconds, free transfer
    interchanges = [
        ("M_Y01", "M_V01", "Kashmere Gate"),        # Yellow <-> Violet
        ("M_Y05", "M_B01", "Rajiv Chowk"),           # Yellow <-> Blue
        ("M_Y07", "M_V08", "Central Secretariat"),    # Yellow <-> Violet
        ("M_V06", "M_B03", "Mandi House"),            # Violet <-> Blue
        ("M_Y14", "M_M01", "Hauz Khas"),              # Yellow <-> Magenta
    ]

    return metro_stations, metro_segments, interchanges


# ───────────────────────────────────────────────────────────
#  FARE CALCULATION
# ───────────────────────────────────────────────────────────

def calculate_metro_fare(distance_km):
    """Delhi Metro fare slabs (2024 rates)."""
    if distance_km <= 2:
        return 10
    elif distance_km <= 5:
        return 20
    elif distance_km <= 12:
        return 30
    elif distance_km <= 21:
        return 40
    elif distance_km <= 32:
        return 50
    else:
        return 60

def calculate_bus_fare(distance_km, ac=False):
    """DTC / cluster bus fare structure."""
    if ac:
        if distance_km <= 4:
            return 10
        elif distance_km <= 10:
            return 15
        else:
            return 25
    else:
        if distance_km <= 4:
            return 5
        elif distance_km <= 10:
            return 10
        else:
            return 15
