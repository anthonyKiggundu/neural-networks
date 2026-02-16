import numpy as np
import os

SEQ_LEN = 24
snapshots = np.load("../data/processed/grid_snapshots.npy", allow_pickle=True).item()
times = sorted(snapshots.keys())

os.makedirs("../data/grid_sequences", exist_ok=True)

seq_id = 0
for i in range(len(times) - SEQ_LEN):
    seq = []
    for t in times[i:i+SEQ_LEN]:
        seq.append(snapshots[t])

    np.savez_compressed(
        f"../data/grid_sequences/seq_{seq_id:06}.npz",
        sequence=seq
    )
    seq_id += 1

print("Built sequences:", seq_id)

