import yaml
import torch
from torch.utils.data import DataLoader

from dataset.dataset import CityWorldDataset
from dataset.sequence_to_batch import SequenceBatchBuilder
from model.world_model import GridCellWorldModel
from training.losses import reconstruction_loss
from experiments.tracking.logger import ExperimentLogger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def run(config_path):
    cfg = load_config(config_path)

    logger = ExperimentLogger(cfg["experiment_name"])
    logger.save_config(cfg)

    dataset = CityWorldDataset("../data/grid_sequences/*.npz")
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)

    model = GridCellWorldModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    builder = SequenceBatchBuilder()

    def sample_events(batch):
        return torch.zeros(batch, cfg["model"]["event_dim"]).to(DEVICE)

    step = 0
    for epoch in range(cfg["training"]["epochs"]):
        # curriculum: grow imagination horizon
        horizon = min(ROLLOUT_HORIZON + epoch // 5, 20)
        print("Rollout horizon:", horizon)

        for seq in loader:
            sparse_seq = seq[0]
            S_t_seq, S_next_seq = builder.build_pair(sparse_seq)

            S_t_seq = S_t_seq.to(DEVICE)
            S_next_seq = S_next_seq.to(DEVICE)

            # ---------- 1 STEP LOSS ----------
            S_t = S_t_seq[0].unsqueeze(0)
            S_next_gt = S_next_seq[0].unsqueeze(0)

            E0 = torch.zeros(1,16).to(DEVICE)
            one_step_pred = model(S_t, E0)

            # ---------- MULTI STEP ROLLOUT ----------
            E_seq = sample_events(1, horizon)
            rollout_pred = rollout_world_model(model, S_t, E_seq, horizon)

            rollout_gt = S_next_seq[:horizon].unsqueeze(0)

            loss = total_loss(one_step_pred, S_next_gt, rollout_pred, rollout_gt)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            logger.log(step, {"loss": total_loss.item()})
            step += 1

        print(f"Epoch {epoch} done")

    logger.save_model(model)

if __name__ == "__main__":
    run("configs/base.yaml")

