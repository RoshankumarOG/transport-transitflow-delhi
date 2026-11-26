# Delhi TransitFlow

Search for a starting stop and destination, or tap two points on the map. The planner automatically compares bus, metro and mixed journeys. Drag either pin to change the trip. Use Journey options to choose a mode, lower estimated fare or fewer changes.

## Run locally

From this checkout in PowerShell, run `.\start.ps1` and open http://127.0.0.1:8001. Run `.\.venv\Scripts\python.exe smoke_test.py` while the server is running to verify routing, input validation and vehicle-feed freshness.

For a new checkout, use Python 3.12, install `requirements.txt`, then download and unpack the Delhi `bus.zip` and `metro-dmrc.zip` archives from [India_GTFS](https://github.com/reclaimchennai/India_GTFS/tree/main/delhi) into `data/real/bus/` and `data/real/metro/`. The GTFS text files belong directly inside each folder. Both feeds are already installed on this computer. Launch `python -m uvicorn app:app --host 127.0.0.1 --port 8001` from the project directory.

## Map and journey data

The local network comes from Delhi Transport Department and DMRC GTFS datasets distributed by reclaim.city. The planner estimates durations from published segment timings, approximate waits and walking distances. Fares use historical distance slabs. The metro feed calendar ended in December 2025, so verify current service with the operator.

Bus markers use the [public vehicle positions feed](https://api.reclaim.city/docs/) with no personal key. The map refreshes every 30 seconds, hides reports older than three minutes, and spaces nearby markers apart at low zoom levels. Select a bus for its route when available and its report time. Tracking, map tiles and Leaflet require internet access. Routing uses the installed network if tracking goes offline.
