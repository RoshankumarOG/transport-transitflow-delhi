"""Public GTFS-RT JSON polling; stale observations never become live markers."""
import json
import math
import time
from threading import Lock
from urllib.request import Request, urlopen

FEED_URL = 'https://maps.reclaim.city/delhi/rt/vehicle_positions.json'
MAX_AGE = 180
_lock = Lock()
_cached = None
_checked = 0

def normalize(payload, now):
    header_time = int(payload.get('header', {}).get('timestamp', 0))
    vehicles = []
    if not -60 <= now - header_time <= MAX_AGE:
        return {'state': 'stale', 'timestamp': header_time, 'vehicles': [], 'message': 'Vehicle feed is delayed. No live positions shown.'}
    for entity in payload.get('entity', []):
        try:
            v = entity['vehicle']; p = v['position']
            timestamp = int(v.get('timestamp', 0))
            lat, lon = float(p['latitude']), float(p['longitude'])
            bearing = float(p.get('bearing', 0))
            if not all(map(math.isfinite, [lat, lon, bearing])) or not (27.8 <= lat <= 29.3 and 76.5 <= lon <= 78.0):
                continue
            if not -60 <= now - timestamp <= MAX_AGE:
                continue
            vehicle = v.get('vehicle', {})
            vehicles.append({'id': str(vehicle.get('id') or entity['id']), 'label': str(vehicle.get('label') or entity['id']),
                             'lat': lat, 'lng': lon, 'bearing': bearing % 360, 'timestamp': timestamp,
                             'route_id': str(v.get('trip', {}).get('routeId', ''))})
        except (KeyError, ValueError, TypeError):
            continue
    return {'state': 'live', 'timestamp': header_time, 'vehicles': vehicles,
            'message': f'{len(vehicles)} buses reporting in the last 3 minutes'}

def vehicle_positions():
    global _cached, _checked
    with _lock:
        now = time.time()
        if _cached is not None and now - _checked < 30:
            return normalize(_cached, now)
        _checked = now
        try:
            request = Request(FEED_URL, headers={'User-Agent': 'TransitFlow/1.0 (local Delhi transit planner)', 'Accept': 'application/json'})
            with urlopen(request, timeout=12) as response:
                payload = response.read(15_000_001)
            if len(payload) > 15_000_000:
                raise ValueError('Feed exceeds size limit')
            _cached = json.loads(payload)
            return normalize(_cached, now)
        except Exception:
            _cached = None
            return {'state': 'unavailable', 'timestamp': None, 'vehicles': [], 'message': 'Live bus feed is unavailable. Route planning still works.'}
