"""Run with the local server on port 8001. No test framework required."""
import json
import re
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from live_data import normalize

BASE = 'http://127.0.0.1:8001'
def call(path, body=None):
    request = Request(BASE + path, data=json.dumps(body).encode() if body is not None else None,
                      headers={'Content-Type':'application/json'})
    with urlopen(request, timeout=120) as response:
        return json.load(response)

def main():
    for path, marker in [('/', b'href="/planner"'), ('/planner', b'src="/planner.js"')]:
        with urlopen(BASE + path) as response:
            assert marker in response.read(), path
    with urlopen(BASE + '/planner') as page, urlopen(BASE + '/planner.js') as script:
        html = page.read().decode()
        required = set(re.findall(r"\$\('([\w-]+)'\)", script.read().decode()))
        missing = required - set(re.findall(r'\bid="([\w-]+)"', html))
        assert not missing, f'Planner elements missing from HTML: {sorted(missing)}'
        assert 'data-from="Connaught Place" data-to="Hauz Khas"' in html
    counts = call('/data-status')
    assert counts['bus_stops'] > 6000 and counts['metro_stations'] > 250
    assert len(call('/stops')) > 6500
    payload = dict(origin_lat=28.6675, origin_lng=77.2282, dest_lat=28.5491, dest_lng=77.2533)
    results = {}
    for mode in ['bus','metro','mixed']:
        for optimize in ['fastest','cheapest','least_transfers']:
            route = call('/plan-journey', dict(payload,mode=mode,optimize=optimize))
            assert 'error' not in route, (mode,optimize,route)
            assert route['total_eta_min'] > 0 and route['fare']['total_fare_min'] >= 0
            used = {l['mode'] for l in route['legs']} - {'walk'}
            assert mode == 'mixed' or used == {mode}, (mode,used)
            assert len(route['legs']) < 100
            if optimize == 'fastest': results[mode] = route['total_eta_min']
    assert results['mixed'] <= min(results['bus'],results['metro']) + .1, results
    assert len(call('/plan-all-options',payload)['options']) == 3
    assert len(call('/metro-lines')) >= 8
    assert 'error' in call('/plan-journey',dict(payload,origin_lat=0,origin_lng=0))
    for change in [{'mode':'invalid'},{'origin_lat':100},{'departure_time':'invalid'}]:
        try:
            call('/plan-journey',dict(payload,**change))
            raise AssertionError('Invalid input accepted')
        except HTTPError as e:
            assert e.code == 422
    feed={'header':{'timestamp':'1000'},'entity':[{'id':'bus','vehicle':{'timestamp':'990','position':{'latitude':28.6,'longitude':77.2}}}]}
    assert len(normalize(feed,1000)['vehicles']) == 1
    assert normalize(feed,1200)['state'] == 'stale'
    feed['entity'][0]['vehicle']['timestamp']='1'
    assert normalize(feed,1000)['vehicles'] == []
    feed['entity'][0]['vehicle']['timestamp']='999'
    feed['entity'][0]['vehicle']['position']['latitude']=float('nan')
    assert normalize(feed,1000)['vehicles'] == []
    live=call('/vehicles')
    assert live['state'] in ['live','stale','unavailable']
    if live['state'] != 'live': assert live['vehicles'] == []
    print('PASS: real network, all modes/preferences, mixed routing, validation, and stale/invalid vehicle filtering')
    print('Public feed:',live['state'],len(live['vehicles']),'fresh vehicles')

if __name__ == '__main__': main()
