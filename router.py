import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from data_pipeline import haversine_m, calculate_metro_fare, calculate_bus_fare

WALK_SPEED_MPS = 1.2            # ~4.3 km/h
BUS_TRANSFER_PENALTY_SEC = 240  # 4 min avg wait for connecting bus
METRO_HEADWAY_SEC = 180         # 3 min avg wait for metro
DWELL_TIME_SEC = 20             # Bus stop boarding dwell
METRO_INTERCHANGE_SEC = 120     # Walk within metro interchange station

class TransitRouter:
    def __init__(self, bus_stops_df, bus_segments_df, metro_stations_df, metro_segments_df, metro_interchanges):
        self.bus_stops_df = bus_stops_df.reset_index(drop=True)
        self.bus_segments_df = bus_segments_df
        self.metro_stations_df = metro_stations_df.reset_index(drop=True)
        self.metro_segments_df = metro_segments_df
        self.metro_interchanges = metro_interchanges

        # Build lookups for stop/station info
        self.stop_lookup = {}
        for _, row in self.bus_stops_df.iterrows():
            self.stop_lookup[row["stop_id"]] = {
                "stop_name": row["stop_name"], "stop_lat": row["stop_lat"], "stop_lon": row["stop_lon"], "mode": "bus"
            }
        for _, row in self.metro_stations_df.iterrows():
            self.stop_lookup[row["station_id"]] = {
                "stop_name": row["station_name"], "stop_lat": row["stop_lat"], "stop_lon": row["stop_lon"], "mode": "metro"
            }

        # Combined spatial index (bus stops + metro stations)
        all_lats = np.concatenate([self.bus_stops_df["stop_lat"].values, self.metro_stations_df["stop_lat"].values])
        all_lons = np.concatenate([self.bus_stops_df["stop_lon"].values, self.metro_stations_df["stop_lon"].values])
        all_ids = np.concatenate([self.bus_stops_df["stop_id"].values, self.metro_stations_df["station_id"].values])
        all_modes = np.array(["bus"] * len(self.bus_stops_df) + ["metro"] * len(self.metro_stations_df))

        self.all_points = np.column_stack([all_lats, all_lons])
        self.all_ids = all_ids
        self.all_modes = all_modes
        self.stop_tree = cKDTree(np.radians(self.all_points))

        # Bus-only and metro-only spatial indices
        self.bus_tree = cKDTree(np.radians(self.bus_stops_df[["stop_lat", "stop_lon"]].values))
        self.metro_tree = cKDTree(np.radians(self.metro_stations_df[["stop_lat", "stop_lon"]].values))

        # Precompute bus-bus transfers
        self.bus_transfer_pairs = self._precompute_bus_transfers(max_walk_m=400)
        # Precompute bus-metro transfers (within 500m)
        self.cross_modal_pairs = self._precompute_cross_modal_transfers(max_walk_m=500)

    def _precompute_bus_transfers(self, max_walk_m=400):
        pairs = self.bus_tree.query_pairs(r=max_walk_m / 6371000.0)
        transfers = []
        for i, j in pairs:
            s1, s2 = self.bus_stops_df.iloc[i], self.bus_stops_df.iloc[j]
            dist = haversine_m(s1["stop_lat"], s1["stop_lon"], s2["stop_lat"], s2["stop_lon"])
            walk_sec = (dist / WALK_SPEED_MPS) + BUS_TRANSFER_PENALTY_SEC
            transfers.append((s1["stop_id"], s2["stop_id"], walk_sec, dist))
            transfers.append((s2["stop_id"], s1["stop_id"], walk_sec, dist))
        return transfers

    def _precompute_cross_modal_transfers(self, max_walk_m=500):
        """Find bus stops within 500m of metro stations for intermodal transfers."""
        transfers = []
        for _, ms in self.metro_stations_df.iterrows():
            dists, idxs = self.bus_tree.query(
                np.radians([[ms["stop_lat"], ms["stop_lon"]]]), k=5
            )
            for dist_rad, idx in zip(dists[0], idxs[0]):
                dist_m = dist_rad * 6371000.0
                if dist_m <= max_walk_m:
                    bs = self.bus_stops_df.iloc[idx]
                    walk_sec = dist_m / WALK_SPEED_MPS
                    transfers.append({
                        "bus_stop_id": bs["stop_id"],
                        "metro_station_id": ms["station_id"],
                        "walk_sec": walk_sec,
                        "distance_m": dist_m
                    })
        return transfers

    def build_graph(self, scored_bus_segments_df, mode="mixed"):
        """
        Builds transit graph.
        mode: 'bus' (bus only), 'metro' (metro only), 'mixed' (bus + metro + intermodal)
        """
        G = nx.DiGraph()

        # ── BUS EDGES ──
        if mode in ("bus", "mixed"):
            for _, row in scored_bus_segments_df.iterrows():
                u = (row["stop_id_from"], row["route_id"])
                v = (row["stop_id_to"], row["route_id"])
                weight = float(row["pred_travel_time_sec"]) + DWELL_TIME_SEC
                G.add_edge(u, v,
                    weight=weight, edge_type="bus", mode="bus",
                    route_id=row["route_id"], route_name=row["route_short_name"],
                    segment_id=row["segment_id"], congestion=row["congestion"],
                    distance_m=float(row["distance_m"])
                )

            # Bus route-to-route transfers at same stop
            stop_routes = scored_bus_segments_df.groupby("stop_id_from")["route_id"].unique().to_dict()
            for stop_id, r_list in scored_bus_segments_df.groupby("stop_id_to")["route_id"].unique().items():
                stop_routes[stop_id] = np.unique(np.concatenate([stop_routes.get(stop_id, []), r_list]))

            for stop_id, r_list in stop_routes.items():
                for r1 in r_list:
                    for r2 in r_list:
                        if r1 != r2:
                            G.add_edge((stop_id, r1), (stop_id, r2),
                                weight=BUS_TRANSFER_PENALTY_SEC, edge_type="bus_transfer",
                                mode="walk", distance_m=0)

            # Bus walking transfers (< 400m)
            for s1, s2, walk_sec, dist in self.bus_transfer_pairs:
                for r1 in stop_routes.get(s1, []):
                    for r2 in stop_routes.get(s2, []):
                        G.add_edge((s1, r1), (s2, r2),
                            weight=walk_sec, edge_type="walk_transfer",
                            mode="walk", distance_m=dist)

        # ── METRO EDGES ──
        if mode in ("metro", "mixed"):
            for _, row in self.metro_segments_df.iterrows():
                u = (row["stop_id_from"], row["route_id"])
                v = (row["stop_id_to"], row["route_id"])
                G.add_edge(u, v,
                    weight=float(row["travel_time_sec"]),
                    edge_type="metro", mode="metro",
                    route_id=row["route_id"], route_name=row["route_short_name"],
                    segment_id=row["segment_id"],
                    congestion="free_flow",
                    distance_m=float(row["distance_m"])
                )

            # Metro line interchanges (walk within station)
            for sid1, sid2, name in self.metro_interchanges:
                lines1 = self.metro_segments_df[
                    (self.metro_segments_df["stop_id_from"] == sid1) | (self.metro_segments_df["stop_id_to"] == sid1)
                ]["route_id"].unique()
                lines2 = self.metro_segments_df[
                    (self.metro_segments_df["stop_id_from"] == sid2) | (self.metro_segments_df["stop_id_to"] == sid2)
                ]["route_id"].unique()

                for l1 in lines1:
                    for l2 in lines2:
                        if l1 != l2:
                            G.add_edge((sid1, l1), (sid2, l2),
                                weight=METRO_INTERCHANGE_SEC, edge_type="metro_interchange",
                                mode="walk", distance_m=0)
                            G.add_edge((sid2, l2), (sid1, l1),
                                weight=METRO_INTERCHANGE_SEC, edge_type="metro_interchange",
                                mode="walk", distance_m=0)

        # ── CROSS-MODAL TRANSFERS (Bus <-> Metro) ──
        if mode == "mixed":
            bus_stop_routes = {}
            if scored_bus_segments_df is not None:
                for stop_id, r_list in scored_bus_segments_df.groupby("stop_id_from")["route_id"].unique().items():
                    bus_stop_routes[stop_id] = r_list
                for stop_id, r_list in scored_bus_segments_df.groupby("stop_id_to")["route_id"].unique().items():
                    bus_stop_routes[stop_id] = np.unique(
                        np.concatenate([bus_stop_routes.get(stop_id, []), r_list]))

            for xfer in self.cross_modal_pairs:
                bus_sid = xfer["bus_stop_id"]
                metro_sid = xfer["metro_station_id"]
                walk_sec = xfer["walk_sec"]
                dist_m = xfer["distance_m"]

                metro_lines = self.metro_segments_df[
                    (self.metro_segments_df["stop_id_from"] == metro_sid) |
                    (self.metro_segments_df["stop_id_to"] == metro_sid)
                ]["route_id"].unique()

                for bus_route in bus_stop_routes.get(bus_sid, []):
                    for metro_line in metro_lines:
                        # Bus → Metro (walk + metro headway wait)
                        G.add_edge((bus_sid, bus_route), (metro_sid, metro_line),
                            weight=walk_sec + METRO_HEADWAY_SEC,
                            edge_type="bus_to_metro", mode="walk", distance_m=dist_m)
                        # Metro → Bus (walk + bus headway wait)
                        G.add_edge((metro_sid, metro_line), (bus_sid, bus_route),
                            weight=walk_sec + BUS_TRANSFER_PENALTY_SEC,
                            edge_type="metro_to_bus", mode="walk", distance_m=dist_m)

        return G

    def snap_point(self, lat, lon, k=5, max_m=1000, mode="mixed"):
        """Snap a lat/lon point to nearest stops/stations."""
        dists, idxs = self.stop_tree.query(np.radians([[lat, lon]]), k=k)
        results = []
        for dist_rad, idx in zip(dists[0], idxs[0]):
            dist_m = dist_rad * 6371000.0
            if dist_m <= max_m:
                stop_id = self.all_ids[idx]
                stop_mode = self.all_modes[idx]
                if mode == "bus" and stop_mode != "bus":
                    continue
                if mode == "metro" and stop_mode != "metro":
                    continue
                results.append((stop_id, dist_m, dist_m / WALK_SPEED_MPS, stop_mode))
        return results

    def _calculate_journey_fare(self, legs):
        """Calculate total fare from journey legs."""
        bus_fare_min = 0
        bus_fare_max = 0
        metro_total_dist_km = 0
        current_bus_dist_m = 0
        current_bus_route = None
        in_metro = False

        for leg in legs:
            if leg["mode"] == "bus":
                if leg.get("route_id") != current_bus_route:
                    # Settle previous bus segment
                    if current_bus_route is not None:
                        d_km = current_bus_dist_m / 1000.0
                        bus_fare_min += calculate_bus_fare(d_km, ac=False)
                        bus_fare_max += calculate_bus_fare(d_km, ac=True)
                    current_bus_route = leg.get("route_id")
                    current_bus_dist_m = leg.get("distance_m", 0)
                else:
                    current_bus_dist_m += leg.get("distance_m", 0)
                in_metro = False

            elif leg["mode"] == "metro":
                metro_total_dist_km += leg.get("distance_m", 0) / 1000.0
                in_metro = True

            elif leg["mode"] == "walk":
                if current_bus_route is not None and not in_metro:
                    d_km = current_bus_dist_m / 1000.0
                    bus_fare_min += calculate_bus_fare(d_km, ac=False)
                    bus_fare_max += calculate_bus_fare(d_km, ac=True)
                    current_bus_route = None
                    current_bus_dist_m = 0

        # Settle last bus ride
        if current_bus_route is not None:
            d_km = current_bus_dist_m / 1000.0
            bus_fare_min += calculate_bus_fare(d_km, ac=False)
            bus_fare_max += calculate_bus_fare(d_km, ac=True)

        metro_fare = calculate_metro_fare(metro_total_dist_km) if metro_total_dist_km > 0 else 0

        return {
            "metro_fare": metro_fare,
            "bus_fare_non_ac": bus_fare_min,
            "bus_fare_ac": bus_fare_max,
            "total_fare_min": metro_fare + bus_fare_min,
            "total_fare_max": metro_fare + bus_fare_max,
            "metro_distance_km": round(metro_total_dist_km, 1)
        }

    def plan_journey(self, G, orig_lat, orig_lng, dest_lat, dest_lng,
                     mode="mixed", optimize="fastest"):
        """
        Plans optimal journey.
        mode: 'bus', 'metro', 'mixed'
        optimize: 'fastest', 'cheapest', 'least_transfers'
        """
        orig_snaps = self.snap_point(orig_lat, orig_lng, mode=mode)
        dest_snaps = self.snap_point(dest_lat, dest_lng, mode=mode)

        if not orig_snaps or not dest_snaps:
            mode_name = 'Bus stops' if mode == 'bus' else 'Metro stations' if mode == 'metro' else 'Stops or stations'
            return {"error": f"No {mode_name} found within 1km of origin or destination."}

        # Weight function tailored to the optimization target
        if optimize == "cheapest":
            def weight_fn(u, v, d):
                edge_mode = d.get("mode", "walk")
                dist_m = d.get("distance_m", 0.0)
                time_s = d.get("weight", 0.0)
                if edge_mode == "walk":
                    cost = 0.0
                elif edge_mode == "bus":
                    # Non-AC Bus is ₹5-₹15, approx ₹1.0 per km + ₹5 boarding penalty
                    is_boarding = 5.0 if "transfer" in d.get("edge_type", "") else 0.0
                    cost = (dist_m / 1000.0) * 1.0 + is_boarding
                elif edge_mode == "metro":
                    # Metro is ₹10-₹60, approx ₹3.5 per km + ₹10 base
                    is_boarding = 10.0 if "interchange" in d.get("edge_type", "") or "to_metro" in d.get("edge_type", "") else 0.0
                    cost = (dist_m / 1000.0) * 3.5 + is_boarding
                else:
                    cost = 0.0
                # Cost is dominant (scaled x1000), time is tiebreaker
                return (cost * 1000.0) + (time_s * 0.05)

        elif optimize == "least_transfers":
            weight_fn = lambda u, v, d: d["weight"] + (3600 if "transfer" in d.get("edge_type", "") or "interchange" in d.get("edge_type", "") else 0)
        else:  # fastest
            weight_fn = "weight"

        # Connect virtual ORIGIN and DESTINATION nodes
        G.add_node("ORIGIN")
        G.add_node("DEST")

        for stop_id, dist_m, walk_sec, stop_mode in orig_snaps:
            initial_wait = 180 if stop_mode == "bus" else METRO_HEADWAY_SEC
            for node in list(G.nodes):
                if isinstance(node, tuple) and node[0] == stop_id:
                    G.add_edge("ORIGIN", node,
                        weight=walk_sec + initial_wait,
                        edge_type=f"walk_to_{stop_mode}",
                        mode="walk", distance_m=dist_m)

        for stop_id, dist_m, walk_sec, stop_mode in dest_snaps:
            for node in list(G.nodes):
                if isinstance(node, tuple) and node[0] == stop_id:
                    G.add_edge(node, "DEST",
                        weight=walk_sec,
                        edge_type=f"walk_from_{stop_mode}",
                        mode="walk", distance_m=dist_m)

        try:
            path = nx.dijkstra_path(G, "ORIGIN", "DEST", weight=weight_fn)
        except nx.NetworkXNoPath:
            G.remove_nodes_from(["ORIGIN", "DEST"])
            return {"error": f"No valid {mode} route found between these points."}

        # Build step-by-step legs
        legs = []
        coordinates = [[orig_lat, orig_lng]]
        total_sec = 0.0

        for u, v in zip(path[:-1], path[1:]):
            edge = G.edges[u, v]
            total_sec += edge["weight"]

            if isinstance(v, tuple):
                s_info = self.stop_lookup.get(v[0], {})
                if s_info:
                    coordinates.append([s_info["stop_lat"], s_info["stop_lon"]])

            edge_mode = edge.get("mode", "walk")
            edge_type = edge.get("edge_type", "")

            # Determine display info
            if edge_mode == "metro":
                label = edge.get("route_name", "Metro")
                icon = "🚇"
            elif edge_mode == "bus":
                label = "Bus " + edge.get("route_name", "")
                icon = "🚌"
            else:
                if "metro" in edge_type:
                    label = "Walk to/from Metro"
                    icon = "🚶"
                elif "bus" in edge_type:
                    label = "Walk to/from Bus"
                    icon = "🚶"
                else:
                    label = "Walk / Transfer"
                    icon = "🚶"

            legs.append({
                "type": edge_type,
                "mode": edge_mode,
                "from_stop": self.stop_lookup[u[0]]["stop_name"] if isinstance(u, tuple) and u[0] in self.stop_lookup else "Origin",
                "to_stop": self.stop_lookup[v[0]]["stop_name"] if isinstance(v, tuple) and v[0] in self.stop_lookup else "Destination",
                "route": label,
                "icon": icon,
                "route_id": edge.get("route_id", None),
                "duration_min": round(edge["weight"] / 60.0, 1),
                "distance_m": edge.get("distance_m", 0),
                "congestion": edge.get("congestion", "none")
            })

        coordinates.append([dest_lat, dest_lng])
        G.remove_nodes_from(["ORIGIN", "DEST"])

        # Calculate fares
        fare_info = self._calculate_journey_fare(legs)
        modes_used = list(set(l["mode"] for l in legs if l["mode"] != "walk"))

        return {
            "total_eta_min": round(total_sec / 60.0, 1),
            "transfers": sum(1 for l in legs if "transfer" in l["type"] or "interchange" in l["type"]),
            "modes_used": modes_used,
            "fare": fare_info,
            "legs": legs,
            "polyline": coordinates,
            "optimize": optimize,
            "mode": mode
        }

    def plan_mode_comparison(self, scored_bus_segments_df, orig_lat, orig_lng, dest_lat, dest_lng, mode="mixed"):
        """
        For the user-selected mode (bus, metro, or mixed), calculates:
        - ⚡ Fastest way
        - 💰 Cheapest way
        """
        results = {}
        for opt in ["fastest", "cheapest"]:
            try:
                G = self.build_graph(scored_bus_segments_df, mode=mode)
                res = self.plan_journey(G, orig_lat, orig_lng, dest_lat, dest_lng, mode=mode, optimize=opt)
                if "error" not in res:
                    label_mode = {"bus": "Bus", "metro": "Metro", "mixed": "Bus + Metro"}.get(mode, mode)
                    res["title"] = f"{'⚡ Fastest' if opt == 'fastest' else '💰 Cheapest'} {label_mode}"
                    res["tag"] = "⚡ Fastest" if opt == "fastest" else "💰 Cheapest"
                    results[opt] = res
            except Exception:
                continue
        return results

    def plan_all_options(self, scored_bus_segments_df, orig_lat, orig_lng, dest_lat, dest_lng):
        """
        Returns fastest routes across Bus, Metro, and Mixed modes with fares and tags.
        """
        options = []
        for plan_mode in ["bus", "metro", "mixed"]:
            try:
                G = self.build_graph(scored_bus_segments_df, mode=plan_mode)
                result = self.plan_journey(G, orig_lat, orig_lng, dest_lat, dest_lng,
                                           mode=plan_mode, optimize="fastest")
                if "error" not in result:
                    result["option_label"] = {
                        "bus": "🚌 Bus Only",
                        "metro": "🚇 Metro Only",
                        "mixed": "🚌+🚇 Bus + Metro"
                    }[plan_mode]
                    options.append(result)
            except Exception:
                continue

        options.sort(key=lambda x: x["total_eta_min"])
        if options:
            options[0]["tags"] = options[0].get("tags", []) + ["⚡ Fastest Overall"]
            cheapest = min(options, key=lambda x: x["fare"]["total_fare_min"])
            cheapest["tags"] = cheapest.get("tags", []) + ["💰 Cheapest Overall"]

        return options
