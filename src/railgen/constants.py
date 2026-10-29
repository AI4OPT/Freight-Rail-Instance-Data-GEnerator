"""Constants not set by user config."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# Where FRIDGE output is saved to
OUTPUT_DIR_NAME = "output"
TRAIN_SCHEDULE_FILE_NAME = "schedule.csv"

# Where input data file are pulled from
INPUT_DATA_DIR = Path("input_data")

# Generated Rail Network Graph and related files
RAIL_NETWORK_DIR = Path("rail_network")

# Reverse geocoding (BigDataCloud)
GEOCODE_API_URL = "https://api-bdc.net/data/reverse-geocode"
GEOCODE_CONCURRENCY = 10  # max simultaneous geocoding requests
BDC_KEY = os.environ.get("BDC_KEY", "")

# Forward geocoding (Nominatim): hub cities and city/state centers of scenario regions.
FORWARD_GEOCODE_API_URL = "https://nominatim.openstreetmap.org/search"
FRIDGE_USER_AGENT = (
    "FRIDGE/1.0 (https://github.com/AI4OPT/Freight-Rail-Instance-Data-GEnerator)"
)
FORWARD_GEOCODE_MIN_INTERVAL_S = 1.05  # Nominatim usage policy: max ~1 request/second

# Graph Generation Settings
# NTAD is on ArcGIS Hub. Each download is a fresh export behind a signed URL that
# expires in about an hour
HUB_DOWNLOAD_URL = "https://hub.arcgis.com/api/download/v1/items/{}/geojson"
YARDS_ITEM_ID = "e8579994d57049a6a702c3fa4af01fcf"  # USDOT_BTS "Rail Yards"
YARDS_FILE = "NTAD_NA_Rail_Yards.geojson"
NA_RAIL_NETWORK_ITEM_ID = (
    "e143f436d4774402aa8cca1e663b1d24"  # USDOT_BTS "North American Rail Network Lines"
)
NA_RAIL_NETWORK_FILE = "NTAD_NA_Rail_Network.geojson"
HUB_EXPORT_POLL_S = 5  # gap between status checks while an export is being generated
HUB_EXPORT_TIMEOUT_S = 900  # give up if the export never reaches "Completed"
# Every NTAD yard record with its elevation and city. The lookups are slow so we cache them here and build `yds_dict.json` from this.
YARD_RECORDS_FILE = "yard_records.json"
# Yards after the changes file is applied
YARDS_DICT_FILE = "yds_dict.json"
# Distance caches, one per grade setting (`build_distance_matrix.distance_cache_file`)
ALL_DISTANCES_GLOB = "all_distances*.json"
GRAPH_FILE = "graph.gexf"
# Ruling grade of each hop of the graph, found along its track when a schedule first needs it. Removed when `graph.gexf` is generated.
EDGE_GRADES_FILE = "edge_grades.json"
# E.g. drop edge A-C if some B has A-B + B-C shorter than A-C, or longer by less than this fraction (likely the same track running through B).
SAME_TRACK_TOLERANCE = 0.03
# Hand fixes to the NTAD yard records. Only rows with apply = `yes` are used.
# - `merge`: duplicate record of a yard, folded into `merge_into`.
# - `drop`: not a freight yard (passenger, tourist) or off the network. Only removed when `graph.gexf` is rebuilt; until then it just can't represent a FAF zone.
# - `bar`: stays in the network but can't represent a FAF zone (probably passenger-only, unconfirmed).
YARD_CHANGES_FILE = "yd_changes.csv"
# Geocoded hub cities (`network_settings.hub_locations`), cached so we don't hit the geocoder every load.
HUB_COORDS_FILE = "hub_coords.json"
# Output of every schedule run: the yards that represent each FAF zone, and which are hubs.
ZONE_YARDS_FILE = "zone_yards.csv"

# Copernicus GLO-90 DEM as 1x1 degree tiles, for the elevation of yards and along the track of a hop.
DEM_TILE_URL = "https://copernicus-dem-90m.s3.amazonaws.com/{name}/{name}.tif"
DEM_DIR = INPUT_DATA_DIR / "copernicus_dem"
DEM_DOWNLOAD_CONCURRENCY = 5  # max simultaneous tile downloads
# The DEM is the ground surface. If we sample over a tunnel or under a bridge we'll get inaccurate exagerrated grades.
MAX_TRACK_GRADE_PERCENT = 4.0
TRACK_OFF_GROUND_M = 10.0

# FAF6 freight data
FAF6_ZIP_URL = "https://faf.ornl.gov/faf6/data/Download_Files/ZIP/FAF6.0.zip"
FAF6_CSV_NAME = "FAF6.0.csv"
# FAF6 zone boundaries (2022 CFS geography shapefile): FAF6.shp, .shx and .dbf are read.
FAF6_ZONES_ZIP_URL = "https://faf.ornl.gov/faf6/data/FAF6.zip"
FAF6_ZONES_DIR_NAME = "FAF6"
FAF6_ZONES_BASE_NAME = "FAF6"
# A yard outside every zone but this close to one belongs to it: a yard's first track point can
# sit just over a zone boundary (EL PASO-193 is 7 m outside El Paso, on the Mexican border).
FAF_ZONE_TOLERANCE_KM = 1.0
# Year of the FAF tons columns: FAF6 has 2022 only. FAF5 ton-miles of the same year give the distances.
FAF_YEAR = "2022"
# FAF5 data - has intra-zone distances which faf6 does not
FAF5_ZIP_URL = "https://faf.ornl.gov/faf5/Data/Download_Files/FAF5.7.1_2018-2024.zip"
FAF5_CSV_NAME = "FAF5.7.1_2018-2024.csv"
