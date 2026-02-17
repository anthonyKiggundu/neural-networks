# src/preprocessing/telecom_rasterizer.py
import numpy as np
from scipy.interpolate import Rbf
from .geo_utils import latlon_to_pixel
from .telecom_features import CHANNELS

class TelecomRasterizer:
    def __init__(self, bounds, H=128, W=128):
        self.bounds = bounds
        self.H = H
        self.W = W
        self.C = len(CHANNELS)

    def rasterize_points(self, df_window):
        """Aggregate point measurements into sparse grid."""
        grid_sum = np.zeros((self.C, self.H, self.W))
        grid_count = np.zeros((self.H, self.W))

        coords_x, coords_y = [], []
        channel_points = {c: [] for c in CHANNELS}

        for _, row in df_window.iterrows():
            i,j = latlon_to_pixel(row.Latitude, row.Longitude, self.bounds, self.H, self.W)

            coords_x.append(i)
            coords_y.append(j)

            for c_idx, c in enumerate(CHANNELS):
                val = row[c]
                grid_sum[c_idx,i,j] += val
                channel_points[c].append(val)

            grid_count[i,j] += 1

        grid_mean = grid_sum / (grid_count + 1e-6)
        return grid_mean, coords_x, coords_y

    def interpolate_dense(self, sparse_grid, coords_x, coords_y):
        """Fill missing cells via RBF interpolation."""
        dense = np.zeros_like(sparse_grid)

        for c in range(self.C):
            values = sparse_grid[c][sparse_grid[c] != 0]
            if len(values) < 10:
                dense[c] = sparse_grid[c]
                continue

            rbfi = Rbf(coords_x, coords_y, values, function='gaussian')
            grid_x, grid_y = np.mgrid[0:self.H, 0:self.W]
            dense[c] = rbfi(grid_x, grid_y)

        return dense

    def rasterize_window(self, df_window):
        sparse, xs, ys = self.rasterize_points(df_window)
        return self.interpolate_dense(sparse, xs, ys)

