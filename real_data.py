"""Published Delhi GTFS network, with no synthetic data fallback."""
from pathlib import Path
import pandas as pd
from data_pipeline import haversine_m

ROOT = Path(__file__).parent / 'data' / 'real'
SOURCE = 'https://github.com/reclaimchennai/India_GTFS/tree/main/delhi'

def load_feed(mode):
    folder = ROOT / mode
    if not (folder / 'stop_times.txt').is_file():
        network = Path(__file__).parent / 'network'
        stops = pd.read_csv(network / f'{mode}-stops.csv.gz')
        segments = pd.read_csv(network / f'{mode}-segments.csv.gz', dtype={'route_short_name': str})
        segments['pred_travel_time_sec'] = segments.travel_time_sec
        segments['congestion'] = 'unknown'
        return stops, segments
    stops = pd.read_csv(folder / 'stops.txt', dtype={'stop_id': str})
    routes = pd.read_csv(folder / 'routes.txt', dtype={'route_id': str}).fillna('')
    trips = pd.read_csv(folder / 'trips.txt', dtype=str).fillna('')
    # Keep each published route pattern, including opposite directions and branches.
    representatives = trips.drop_duplicates(['route_id', 'shape_id'] if 'shape_id' in trips else ['route_id'])
    times = pd.read_csv(folder / 'stop_times.txt', dtype={'trip_id': str, 'stop_id': str})
    times = times[times.trip_id.isin(representatives.trip_id)].merge(representatives[['trip_id', 'route_id']], on='trip_id')
    times = times.sort_values(['trip_id', 'stop_sequence'])
    times['next_stop_id'] = times.groupby('trip_id').stop_id.shift(-1)
    def seconds(values):
        parts = values.str.split(':', expand=True).astype(float)
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    times['seconds'] = seconds(times.arrival_time)
    times['travel_time_sec'] = times.groupby('trip_id').seconds.shift(-1) - seconds(times.departure_time)
    segments = times.dropna(subset=['next_stop_id']).copy()
    segments = segments.groupby(['route_id', 'stop_id', 'next_stop_id'], as_index=False).travel_time_sec.median()
    segments = segments.merge(routes[['route_id', 'route_short_name', 'route_long_name']], on='route_id')
    cols = ['stop_id', 'stop_name', 'stop_lat', 'stop_lon']
    segments = segments.merge(stops[cols], on='stop_id').merge(stops[cols], left_on='next_stop_id', right_on='stop_id', suffixes=('_from', '_to'))
    segments['distance_m'] = haversine_m(segments.stop_lat_from, segments.stop_lon_from, segments.stop_lat_to, segments.stop_lon_to)
    # Published bus timings themselves are estimates; do not label these as live ETAs.
    fallback = segments.distance_m / (18 / 3.6 if mode == 'bus' else 32 / 3.6)
    segments['travel_time_sec'] = segments.travel_time_sec.where(segments.travel_time_sec > 0, fallback).clip(lower=15)
    for col in ['stop_id_from', 'stop_id_to']:
        segments[col] = mode + ':' + segments[col]
    stops['stop_id'] = mode + ':' + stops.stop_id
    if mode == 'metro':
        segments['route_short_name'] = 'Metro ' + segments.route_long_name.str.split('_').str[0].str.title()
    segments['route_id'] = mode + ':' + segments.route_id
    segments['segment_id'] = segments.route_id + '_' + segments.stop_id_from + '_' + segments.stop_id_to
    segments['pred_travel_time_sec'] = segments.travel_time_sec
    segments['pred_speed_kmph'] = segments.distance_m / segments.travel_time_sec * 3.6
    segments['congestion'] = 'unknown'
    active = set(segments.stop_id_from) | set(segments.stop_id_to)
    stops = stops[stops.stop_id.isin(active)].copy()
    return stops, segments

def load_network():
    bus_stops, bus_segments = load_feed('bus')
    metro_stops, metro_segments = load_feed('metro')
    metro_stops = metro_stops.rename(columns={'stop_id':'station_id','stop_name':'station_name'})
    membership = {}
    for s in metro_segments.itertuples():
        for sid in [s.stop_id_from, s.stop_id_to]:
            membership.setdefault(sid, set()).add(s.route_short_name.replace('Metro ', ''))
    metro_stops['line'] = metro_stops.station_id.map(lambda sid: ' / '.join(sorted(membership.get(sid, []))))
    interchanges = [(s.station_id, s.station_id, s.station_name) for s in metro_stops.itertuples() if len(membership[s.station_id]) > 1]
    return bus_stops, bus_segments, metro_stops, metro_segments, interchanges
