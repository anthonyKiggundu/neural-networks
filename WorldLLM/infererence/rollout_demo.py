import torch
from model.world_model import GridCellWorldModel
from validation.validator import WorldModelValidator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = GridCellWorldModel().to(DEVICE)
model.load_state_dict(torch.load("../checkpoints/world_model.pt"))

validator = WorldModelValidator(model)

# dummy current state
S_t = torch.rand(1,4,128,128).to(DEVICE)
E_t = torch.zeros(1,16).to(DEVICE)

metrics = validator.run_counterfactual_rollout(S_t, E_t)

print("Simulation metrics:", metrics)

