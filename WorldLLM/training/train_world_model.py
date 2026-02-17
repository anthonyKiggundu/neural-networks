import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from pathlib import Path

from model.world_model import GridCellWorldModel

# ------------------------
# Dataset for Multi-Step Rollout
# ------------------------
DATA_PATH = "data/processed/world_tensor.npy"

class WorldTensorDataset(Dataset):
    """
    Returns sequences for multi-step rollout training:
        context: [T_context, C, H, W]
        future:  [T_future, C, H, W]
    """
    def __init__(self, context_len=12, future_len=6):
        tensor = np.load(DATA_PATH)  # [T, C, H, W]
        self.data = torch.tensor(tensor, dtype=torch.float32)
        self.context_len = context_len
        self.future_len = future_len
        self.num_samples = len(self.data) - (context_len + future_len)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        t0 = idx
        t1 = idx + self.context_len
        t2 = t1 + self.future_len
        context = self.data[t0:t1]  # [T_context, C, H, W]
        future = self.data[t1:t2]   # [T_future, C, H, W]
        return context, future

# ------------------------
# Rollout Loss Function
# ------------------------
def rollout_loss(model, context, events=None):
    """
    Autoregressive multi-step rollout.
    Teacher-forcing only on first step.
    
    Args:
        context: [B, T_context, C, H, W]
        events:  optional [B, event_dim] (can be None)
    """
    B, T_ctx, C, H, W = context.shape
    T_future = context.shape[1]  # For simplicity, using same as context_len if needed

    # Use last observed state as initial step
    state = context[:, -1]  # [B, C, H, W]
    loss = 0

    # Predict rollout horizon
    for t in range(T_future):
        if events is not None:
            pred = model(state, events)
        else:
            pred = model(state, torch.zeros(B, 16))  # dummy events
        target = context[:, t]  # For simplicity, align lengths
        loss += F.mse_loss(pred, target)
        state = pred  # autoregressive
    return loss / T_future

# ------------------------
# Training Loop
# ------------------------
def train_world_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", DEVICE)

    # Dataset & loader
    dataset = WorldTensorDataset(context_len=12, future_len=6)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)  # reduce workers for small memory

    # Sample dimensions
    sample_ctx, _ = dataset[0]
    C = sample_ctx.shape[1]
    H = sample_ctx.shape[2]
    W = sample_ctx.shape[3]

    # Initialize model
    model = GridCellWorldModel(H=H, W=W, in_channels=C)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    epochs = 20

    # Create checkpoints folder
    Path("checkpoints").mkdir(exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}")

        for context, future in pbar:
            context = context.to(DEVICE)        # [B, T, C, H, W]
            future = future.to(DEVICE)          # same shape
            B, T, C, H, W = context.shape

            optimizer.zero_grad()
            loss = rollout_loss(model, context)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch} avg loss:", epoch_loss / len(loader))
        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoints/gc_world_model_epoch{epoch}.pt")

if __name__ == "__main__":
    train_world_model()
