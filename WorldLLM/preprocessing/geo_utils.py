# src/preprocessing/geo_utils.py
import numpy as np

def compute_bounds(df, padding=0.01):
    return (
        df["Latitude"].min() - padding,
        df["Longitude"].min() - padding,
        df["Latitude"].max() + padding,
        df["Longitude"].max() + padding,
    )

def latlon_to_pixel(lat, lon, bounds, H, W):
    min_lat, min_lon, max_lat, max_lon = bounds

    x = (lon - min_lon) / (max_lon - min_lon) * (W - 1)
    y = (lat - min_lat) / (max_lat - min_lat) * (H - 1)

    return int(np.clip(y,0,H-1)), int(np.clip(x,0,W-1))

