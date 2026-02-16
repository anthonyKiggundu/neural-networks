import pandas as pd
import numpy as np
from pyproj import Transformer

INPUT = "../data/raw/nperf_kaiserslautern.csv"
OUTPUT = "../data/processed/observations.parquet"

CELL_SIZE = 100  # meters grid size

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

OP_MAP = {"Telekom":0, "Vodafone":1, "O2":2}
TECH_MAP = {"4G":0, "5G":1}

def encode_time(ts):
    dt = pd.to_datetime(ts, unit="s")
    tod = dt.hour + dt.minute/60
    dow = dt.dayofweek

    return {
        "tod_sin": np.sin(2*np.pi*tod/24),
        "tod_cos": np.cos(2*np.pi*tod/24),
        "dow_sin": np.sin(2*np.pi*dow/7),
        "dow_cos": np.cos(2*np.pi*dow/7),
    }

def latlon_to_grid(lat, lon):
    x_m, y_m = transformer.transform(lon, lat)
    return int(x_m//CELL_SIZE), int(y_m//CELL_SIZE)

df = pd.read_csv(INPUT)

rows = []
for r in df.itertuples():
    x, y = latlon_to_grid(r.lat, r.lon)
    t = encode_time(r.timestamp)

    rows.append({
        "timestamp": r.timestamp,
        "x": x, "y": y,
        "operator_id": OP_MAP.get(r.operator,0),
        "tech_id": TECH_MAP.get(r.technology,0),
        "download": r.download,
        "upload": r.upload,
        "latency": r.latency,
        **t
    })

pd.DataFrame(rows).to_parquet(OUTPUT)
print("Saved observations.parquet")

