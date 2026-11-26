from datetime import datetime, timezone, timedelta
from pathlib import Path
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing import Literal
from real_data import load_network, SOURCE
from live_data import vehicle_positions
from router import TransitRouter

app = FastAPI(title='Delhi TransitFlow — real network and live buses')
ROOT = Path(__file__).parent
transit_router = None
segments_df = metro_stations_df = metro_segments_df = None
COLORS = {'Yellow':'#c5aa29','Blue':'#578ebc','Violet':'#9071a9','Magenta':'#bd699d','Red':'#cc5855','Green':'#548761','Pink':'#df83a3','Gray':'#777777','Orange':'#da8a39','Rapid':'#489fbb'}

@app.on_event('startup')
def startup_event():
    global transit_router, segments_df, metro_stations_df, metro_segments_df
    if transit_router is not None:
        return
    stops, segments_df, metro_stations_df, metro_segments_df, interchanges = load_network()
    transit_router = TransitRouter(stops, segments_df, metro_stations_df, metro_segments_df, interchanges)

@app.get('/')
def read_root():
    return FileResponse(ROOT / 'landing.html')

@app.get('/landing-map')
def landing_map():
    return RedirectResponse('https://www.openstreetmap.org/export/embed.html?bbox=77.13,28.58,77.31,28.69&layer=mapnik')

@app.get('/planner')
def planner_page():
    return FileResponse(ROOT / 'index.html')

class JourneyRequest(BaseModel):
    origin_lat: float = Field(ge=-90, le=90)
    origin_lng: float = Field(ge=-180, le=180)
    dest_lat: float = Field(ge=-90, le=90)
    dest_lng: float = Field(ge=-180, le=180)
    departure_time: datetime = Field(default_factory=lambda: datetime.now(timezone(timedelta(hours=5, minutes=30))))
    mode: Literal['bus', 'metro', 'mixed'] = 'mixed'
    optimize: Literal['fastest', 'cheapest', 'least_transfers'] = 'fastest'

@lru_cache(maxsize=3)
def graph(mode):
    return transit_router.build_graph(segments_df, mode)

def journey(req, mode=None, optimize=None):
    mode = mode or req.mode
    result = transit_router.plan_journey(graph(mode), req.origin_lat, req.origin_lng, req.dest_lat, req.dest_lng,
                                         mode=mode, optimize=optimize or req.optimize)
    if 'error' not in result:
        result['estimate_notice'] = 'Published network; approximate travel time and historical fare tables. Not a confirmed departure.'
        # Collapse adjacent ride segments so a 20-stop train ride is one instruction.
        legs = []
        for leg in result['legs']:
            if leg['duration_min'] == 0 and leg['mode'] == 'walk':
                continue
            if legs and leg['mode'] != 'walk' and legs[-1]['route_id'] == leg['route_id'] and legs[-1]['mode'] == leg['mode']:
                legs[-1]['to_stop'] = leg['to_stop']
                legs[-1]['duration_min'] = round(legs[-1]['duration_min'] + leg['duration_min'], 1)
            else:
                legs.append(leg.copy())
        result['legs'] = legs
    return result

@app.post('/plan-journey')
def plan_journey(req: JourneyRequest):
    return journey(req)

@app.post('/plan-mode-comparison')
def plan_mode_comparison(req: JourneyRequest):
    return {opt: result for opt in ['fastest','cheapest','least_transfers'] if 'error' not in (result := journey(req, optimize=opt))}

@app.post('/plan-all-options')
def plan_all_options(req: JourneyRequest):
    options = [result for mode in ['bus','metro','mixed'] if 'error' not in (result := journey(req, mode=mode))]
    return {'options': sorted(options, key=lambda r:r['total_eta_min'])}

@app.get('/stops')
def stops():
    return [{'id':str(sid), 'name':s['stop_name'], 'lat':s['stop_lat'], 'lng':s['stop_lon'], 'mode':s['mode']} for sid,s in transit_router.stop_lookup.items()]

@app.get('/data-status')
def data_status():
    return {'source':SOURCE, 'bus_stops':len(transit_router.bus_stops_df), 'metro_stations':len(metro_stations_df),
            'bus_routes':int(segments_df.route_id.nunique()), 'metro_notice':'Metro feed calendar ended December 2025; used for network connections, not current departures.',
            'timing_notice':'Bus feed times are approximate. Journey durations and fares are estimates, not live arrival predictions.'}

@app.get('/vehicles')
def vehicles():
    data = vehicle_positions()
    names = dict(zip(segments_df.route_id, segments_df.route_short_name))
    for v in data['vehicles']:
        v['route_name'] = str(names.get('bus:'+v['route_id'], 'Unmatched route'))
    return data

@app.get('/metro-lines')
def metro_lines():
    lines=[]
    for name, group in metro_segments_df.groupby('route_short_name'):
        lines.append({'name':name, 'color':COLORS.get(name.replace('Metro ',''),'#789081'), 'segments':[
            {'from':[s.stop_lat_from,s.stop_lon_from], 'to':[s.stop_lat_to,s.stop_lon_to], 'station_from':s.stop_name_from,'station_to':s.stop_name_to}
            for s in group.drop_duplicates(['stop_id_from','stop_id_to']).itertuples()]})
    return lines

@app.get('/metro-stations')
def metro_stations():
    return metro_stations_df[['station_id','station_name','stop_lat','stop_lon','line']].to_dict(orient='records')

@app.get('/segments-status')
def segments_status():
    return []  # No measured segment traffic feed: do not invent congestion colours.

@app.get('/planner.js')
def planner_script():
    return FileResponse(ROOT / 'planner.js', media_type='text/javascript')
