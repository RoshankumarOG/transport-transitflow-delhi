import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from data_pipeline import haversine_m

WALK_SPEED_MPS = 1.2       # ~4.3 km/h
TRANSFER_PENALTY_SEC = 240  # 4 min avg wait for connecting bus
DWELL_TIME_SEC = 20         # Avg bus stop boarding dwell time

class TransitRouter:
    def __init__(self, stops_df, segments_df):
        self.stops_df = stops_df.reset_index(drop=True)
        self.segments_df = segments_df
        self.stop_lookup = self.stops_df.set_index("stop_id").to_dict("index")

        # Spatial index for fast origin/destination snapping & transfer edges
        self.stop_tree = cKDTree(np.radians(self.stops_df[["stop_lat", "stop_lon"]].values))
        self.transfer_pairs = self._precompute_transfers(max_walk_m=400)

    def _precompute_transfers(self, max_walk_m=400):
        pairs = self.stop_tree.query_pairs(r=max_walk_m / 6371000.0)
        transfers = []
        for i, j in pairs:
            s1, s2 = self.stops_df.iloc[i], self.stops_df.iloc[j]
            dist = haversine_m(s1["stop_lat"], s1["stop_lon"], s2["stop_lat"], s2["stop_lon"])
            walk_sec = (dist / WALK_SPEED_MPS) + TRANSFER_PENALTY_SEC
            transfers.append((s1["stop_id"], s2["stop_id"], walk_sec, dist))
            transfers.append((s2["stop_id"], s1["stop_id"], walk_sec, dist))
        return transfers

    def build_graph(self, scored_segments_df):
        G = nx.DiGraph()

        # Add transit edges (state node = (stop_id, route_id) to track route transfers)
        for _, row in scored_segments_df.iterrows():
            u = (row["stop_id_from"], row["route_id"])
            v = (row["stop_id_to"], row["route_id"])
            weight = float(row["pred_travel_time_sec"]) + DWELL_TIME_SEC
            G.add_edge(
                u, v,
                weight=weight,
                edge_type="bus",
                route_id=row["route_id"],
                route_name=row["route_short_name"],
                segment_id=row["segment_id"],
                congestion=row["congestion"],
                distance_m=float(row["distance_m"])
            )

        # Map which routes serve each stop
        stop_routes = scored_segments_df.groupby("stop_id_from")["route_id"].unique().to_dict()
        for stop_id, r_list in scored_segments_df.groupby("stop_id_to")["route_id"].unique().items():
            stop_routes[stop_id] = np.unique(np.concatenate([stop_routes.get(stop_id, []), r_list]))

        # Same-stop route switches
        for stop_id, r_list in stop_routes.items():
            for r1 in r_list:
                for r2 in r_list:
                    if r1 != r2:
                        G.add_edge((stop_id, r1), (stop_id, r2), weight=TRANSFER_PENALTY_SEC, edge_type="transfer", distance_m=0)

        # Walking transfers between nearby stops (< 400m)
        for s1, s2, walk_sec, dist in self.transfer_pairs:
            for r1 in stop_routes.get(s1, []):
                for r2 in stop_routes.get(s2, []):
                    G.add_edge((s1, r1), (s2, r2), weight=walk_sec, edge_type="walk_transfer", distance_m=dist)

        return G

    def snap_point(self, lat, lon, k=3, max_m=1000):
        dists, idxs = self.stop_tree.query(np.radians([[lat, lon]]), k=k)
        results = []
        for dist_rad, idx in zip(dists[0], idxs[0]):
            dist_m = dist_rad * 6371000.0
            if dist_m <= max_m:
                stop_id = self.stops_df.iloc[idx]["stop_id"]
                results.append((stop_id, dist_m, dist_m / WALK_SPEED_MPS))
        return results

    def plan_journey(self, G, orig_lat, orig_lng, dest_lat, dest_lng):
        orig_snaps = self.snap_point(orig_lat, orig_lng)
        dest_snaps = self.snap_point(dest_lat, dest_lng)

        if not orig_snaps or not dest_snaps:
            return {"error": "No bus stops found within 1km of origin or destination."}

        # Connect virtual ORIGIN and DESTINATION nodes
        G.add_node("ORIGIN")
        G.add_node("DEST")

        for stop_id, dist_m, walk_sec in orig_snaps:
            for node in list(G.nodes):
                if isinstance(node, tuple) and node[0] == stop_id:
                    G.add_edge("ORIGIN", node, weight=walk_sec + 180, edge_type="walk_origin", distance_m=dist_m)

        for stop_id, dist_m, walk_sec in dest_snaps:
            for node in list(G.nodes):
                if isinstance(node, tuple) and node[0] == stop_id:
                    G.add_edge(node, "DEST", weight=walk_sec, edge_type="walk_dest", distance_m=dist_m)

        try:
            path = nx.dijkstra_path(G, "ORIGIN", "DEST", weight="weight")
        except nx.NetworkXNoPath:
            G.remove_nodes_from(["ORIGIN", "DEST"])
            return {"error": "No valid bus route found between these points."}

        # Format step-by-step instructions and map coordinates
        legs = []
        coordinates = [[orig_lat, orig_lng]]
        total_sec = 0.0

        for u, v in zip(path[:-1], path[1:]):
            edge = G.edges[u, v]
            total_sec += edge["weight"]

            if isinstance(v, tuple):
                s_info = self.stop_lookup[v[0]]
                coordinates.append([s_info["stop_lat"], s_info["stop_lon"]])

            legs.append({
                "type": edge["edge_type"],
                "from_stop": self.stop_lookup[u[0]]["stop_name"] if isinstance(u, tuple) else "Origin",
                "to_stop": self.stop_lookup[v[0]]["stop_name"] if isinstance(v, tuple) else "Destination",
                "route": edge.get("route_name", "Walk"),
                "duration_min": round(edge["weight"] / 60.0, 1),
                "congestion": edge.get("congestion", "none")
            })

        coordinates.append([dest_lat, dest_lng])
        G.remove_nodes_from(["ORIGIN", "DEST"])

        return {
            "total_eta_min": round(total_sec / 60.0, 1),
            "transfers": sum(1 for l in legs if "transfer" in l["type"]),
            "legs": legs,
            "polyline": coordinates
        }
