import torch
from torch.utils.data import DataLoader
from dataset import CityWorldDataset
from sequence_to_batch import SequenceBatchBuilder
from model import GridCellWorldModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CityWorldDataset()
loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = GridCellWorldModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
builder = SequenceBatchBuilder()

def sample_events(batch_size):
    # For now: only time-of-day embedding
    return torch.zeros(batch_size, 16).to(DEVICE)

for epoch in range(20):
    for seq in loader:

        inputs_sparse = seq[0]
        S_t_seq, S_next_seq = builder.build_pair(inputs_sparse)

        S_t_seq = S_t_seq.to(DEVICE)
        S_next_seq = S_next_seq.to(DEVICE)

        loss_total = 0

        for t in range(S_t_seq.shape[0]):
            S_t = S_t_seq[t].unsqueeze(0)
            S_next_gt = S_next_seq[t].unsqueeze(0)

            events = sample_events(1)

            S_pred = model(S_t, events)

            loss = torch.nn.functional.mse_loss(S_pred, S_next_gt)
            loss_total += loss

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

    print("epoch", epoch, "loss", loss_total.item())

