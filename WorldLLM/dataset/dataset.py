import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

DATA_PATH = "data/processed/world_tensor.npy"


class WorldModelDataset(Dataset):
    """
    Produces training samples for multi-step rollout training.

    Input  : [T_context, num_cells, num_features]
    Target : [T_future,  num_cells, num_features]
    """

    def __init__(
        self,
        context_len=12,   # past timesteps
        future_len=6      # rollout horizon
    ):
        super().__init__()

        self.context_len = context_len
        self.future_len = future_len

        print("Loading world dataset...")
        df = pd.read_parquet(DATA_PATH)

        # dataset expected format from build_world_dataset:
        # columns: timestamp, cell_id, f1..fN
        feature_cols = [c for c in df.columns if c not in ["timestamp", "cell_id"]]

        # pivot into [time, cell, feature]
        pivot = df.pivot_table(
            index="timestamp",
            columns="cell_id",
            values=feature_cols
        )

        # reorder multiindex -> [time, cell, feature]
        pivot = pivot.sort_index()
        times = pivot.index.values

        num_times = len(times)
        num_cells = len(pivot.columns.levels[1])
        num_features = len(feature_cols)

        data = pivot.values.reshape(num_times, num_cells, num_features)
        self.data = torch.tensor(data, dtype=torch.float32)

        self.num_samples = num_times - (context_len + future_len)
        print("Dataset samples:", self.num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        t0 = idx
        t1 = idx + self.context_len
        t2 = t1 + self.future_len

        context = self.data[t0:t1]  # [T_context, cells, features]
        future = self.data[t1:t2]   # [T_future, cells, features]

        return context, future
