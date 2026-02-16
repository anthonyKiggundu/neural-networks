import pandas as pd

BIN = 15 * 60  # 15 minutes

df = pd.read_parquet("../data/processed/observations.parquet")
df["time_bin"] = df.timestamp // BIN

df.to_parquet("../data/processed/observations_binned.parquet")
print("Saved observations_binned.parquet")

