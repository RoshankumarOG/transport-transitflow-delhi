# Delhi TransitFlow

**One map to decide how to cross Delhi.** Choose two stops and compare a bus journey, a metro journey and a combination of both. See how much time, money and walking each option could take, then tap a bus on the map to check its last reported position.

## The problem

A commuter deciding how to get from Kashmiri Gate to Nehru Place has two different questions: which route fits the trip, and where are the buses right now? TransitFlow puts the route comparison and the vehicle reports on the same map. Its local dataset covers **6,812 bus stops, 2,554 bus routes and 262 metro stations**.

## One-minute demo

1. Open the planner and choose **Kashmiri Gate → Nehru Place** under **Try a journey**. One click fills both stops and starts the comparison.
2. Select **View routes**. Compare the estimated minutes, fare range and number of changes. Select a card to follow its route on the map and read the boarding and transfer instructions.
3. Zoom in and select a bus marker. Its popup shows the vehicle, the route when matched, and the age of its position report. Drag pin **B** to try another destination.

## How it works

| Part | What it does |
| --- | --- |
| Delhi GTFS datasets | Supply stops, route patterns and segment timings. Pandas reads the installed bus and metro feeds at startup. |
| FastAPI and NetworkX | Connect transit legs with walking transfers and compare paths by duration, estimated fare or changes. |
| Leaflet and JavaScript | Draw the selected journey and fetch public bus positions every 30 seconds. Markers expire after three minutes. |

Network connections come from published transit data. Journey times include estimated walking and waiting, and fares use historical distance slabs. The metro feed’s timetable ended in December 2025. Use it to explore metro connections and check current departures with the operator. [Delhi feed provenance](https://github.com/reclaimchennai/India_GTFS/tree/main/delhi) · [Live feed documentation](https://api.reclaim.city/docs/)

## Run the demo locally

On the prepared Windows checkout, run `./start.ps1` and open [the planner](http://127.0.0.1:8001). For a fresh checkout, use Python 3.12 and run these commands from the repository directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$source = 'https://raw.githubusercontent.com/reclaimchennai/India_GTFS/main/delhi'
foreach ($feed in @{bus='bus.zip'; metro='metro-dmrc.zip'}.GetEnumerator()) {
    $folder = "data/real/$($feed.Key)"
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
    $archive = "data/real/$($feed.Value)"
    Invoke-WebRequest "$source/$($feed.Value)" -OutFile $archive
    Expand-Archive -LiteralPath $archive -DestinationPath $folder -Force
}
.\start.ps1
```

No personal API key is required. Live tracking and map tiles need internet access. The routing service can continue using its installed datasets during a tracking outage.

## Verify the demo

With the server running, run `.\.venv\Scripts\python.exe smoke_test.py` from another terminal. It checks the real network counts, all three modes and preferences, invalid inputs, and the expiry of stale vehicle reports. The [local API explorer](http://127.0.0.1:8001/docs) lets you inspect the responses used by the map.

## Maps

Open `/` to explore central Delhi or `/planner` to compare journeys. Both maps use OpenStreetMap and require no API key. The planner displays routes, stops and vehicle reports using Leaflet.
