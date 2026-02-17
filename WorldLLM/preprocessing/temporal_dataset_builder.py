# src/preprocessing/temporal_dataset_builder.py
import pandas as pd
import numpy as np
from tqdm import tqdm
from .telecom_rasterizer import TelecomRasterizer

class TemporalWorldBuilder:
    def __init__(self, bounds, H=128, W=128, window="5min"):
        self.rasterizer = TelecomRasterizer(bounds, H, W)
        self.window = window

    def build_tensor_sequence(self, df):
        df = df.sort_values("timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        groups = df.groupby(pd.Grouper(key="timestamp", freq=self.window))

        frames = []
        times = []

        for t, group in tqdm(groups):
            if len(group) < 50:
                continue
            grid = self.rasterizer.rasterize_window(group)
            frames.append(grid)
            times.append(t)

        return np.stack(frames), times

