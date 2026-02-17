# src/preprocessing/build_world_dataset.py
import pandas as pd
import numpy as np
from pathlib import Path

from .telecom_features import engineer_features
from .geo_utils import compute_bounds
from .temporal_dataset_builder import TemporalWorldBuilder

RAW_PATH = "data/raw/telecom_kpi.csv"
OUT_PATH = "data/processed/world_tensor.npy"

def main():
    print("Loading raw KPI dataset...")
    df = pd.read_csv(RAW_PATH)

    print("Engineering telecom features...")
    df = engineer_features(df)

    print("Computing spatial bounds...")
    bounds = compute_bounds(df)

    print("Building spatio-temporal tensor sequence...")
    builder = TemporalWorldBuilder(bounds)
    tensor_seq, timestamps = builder.build_tensor_sequence(df)

    print("Saving dataset...")
    Path("data/processed").mkdir(exist_ok=True, parents=True)
    np.save(OUT_PATH, tensor_seq)

    print("Saved world tensor:", tensor_seq.shape)

if __name__ == "__main__":
    main()

