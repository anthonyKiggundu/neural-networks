import pandas as pd
import numpy as np
from collections import defaultdict

df = pd.read_parquet("../data/processed/observations_binned.parquet")

snapshots = defaultdict(list)

for r in df.itertuples():
    vec = np.array([
        r.tod_sin, r.tod_cos,
        r.dow_sin, r.dow_cos,
        r.operator_id,
        r.tech_id,
        r.download,
        r.upload,
        r.latency
    ], dtype=np.float32)

    snapshots[r.time_bin].append((r.x, r.y, vec))

np.save("../data/processed/grid_snapshots.npy", snapshots)
print("Saved grid_snapshots.npy")

