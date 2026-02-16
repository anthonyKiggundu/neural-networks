import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
from pyproj import Transformer

class CityWorldDataset(Dataset):
    def __init__(self, path="grid_sequences/*.npz"):
        self.files = glob.glob(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        seq = data["sequence"]

        # convert sparse obs → tensors
        processed_seq = []
        for timestep in seq:
            coords = []
            feats = []
            for (x,y,v) in timestep:
                coords.append([x,y])
                feats.append(v)

            coords = torch.tensor(coords, dtype=torch.long)
            feats = torch.tensor(feats, dtype=torch.float32)
            processed_seq.append((coords, feats))

        return processed_seq


CELL_SIZE = 100  # meters

# lat/lon → metric projection
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")

def encode_time(ts):
    dt = pd.to_datetime(ts, unit="s")
    tod = dt.hour + dt.minute/60
    dow = dt.dayofweek

    tod_sin = np.sin(2*np.pi*tod/24)
    tod_cos = np.cos(2*np.pi*tod/24)
    dow_sin = np.sin(2*np.pi*dow/7)
    dow_cos = np.cos(2*np.pi*dow/7)

    return tod_sin, tod_cos, dow_sin, dow_cos

def latlon_to_grid(lat, lon):
    x_m, y_m = transformer.transform(lat, lon)
    return int(x_m // CELL_SIZE), int(y_m // CELL_SIZE)


def save_snapshots():
    import numpy as np
    import pandas as pd
    from collections import defaultdict

    df = pd.read_parquet("processed/observations_binned.parquet")

    timesteps = sorted(df.time_bin.unique())

    snapshots = defaultdict(list)

    for t in timesteps:
        chunk = df[df.time_bin == t]

        for r in chunk.itertuples():
            obs_vec = np.array([
                r.tod_sin, r.tod_cos,
                r.dow_sin, r.dow_cos,
                r.operator_id,
                r.tech_id,
                r.download,
                r.upload,
                r.latency
            ], dtype=np.float32)

            snapshots[t].append((r.x, r.y, obs_vec))

    np.save("processed/grid_snapshots.npy", snapshots)



def main()

    df = pd.read_csv("raw/nperf.csv")

    rows = []

    for r in df.itertuples():
        x, y = latlon_to_grid(r.lat, r.lon)
        tod_sin, tod_cos, dow_sin, dow_cos = encode_time(r.timestamp)

        rows.append({
            "timestamp": r.timestamp,
            "x": x, "y": y,
            "tod_sin": tod_sin, "tod_cos": tod_cos,
            "dow_sin": dow_sin, "dow_cos": dow_cos,
            "operator_id": OP_MAP[r.operator],
            "tech_id": TECH_MAP[r.technology],
            "download": r.download,
            "upload": r.upload,
            "latency": r.latency,
        })

    pd.DataFrame(rows).to_parquet("processed/observations.parquet")

    import pandas as pd

    df = pd.read_parquet("processed/observations.parquet")

    BIN = 15 * 60  # seconds
    df["time_bin"] = df.timestamp // BIN

    df.to_parquet("processed/observations_binned.parquet")


    snapshots = np.load("processed/grid_snapshots.npy", allow_pickle=True).item()

    SEQ_LEN = 24
    times = sorted(snapshots.keys())

    seq_id = 0

    for i in range(len(times) - SEQ_LEN):
        seq = []

        for t in times[i:i+SEQ_LEN]:
            seq.append(snapshots[t])

        np.savez_compressed(
            f"grid_sequences/seq_{seq_id:06}.npz",
            sequence=seq
        )
        seq_id += 1

