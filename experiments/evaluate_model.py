import torch
from model.world_model import GridCellWorldModel
from validation.validator import WorldModelValidator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(model_path):
    model = GridCellWorldModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    validator = WorldModelValidator(model)

    S_t = torch.rand(1,4,128,128).to(DEVICE)
    E_t = torch.zeros(1,16).to(DEVICE)

    metrics = validator.run_counterfactual_rollout(S_t, E_t, horizon=50)

    print("Evaluation metrics:")
    for k,v in metrics.items():
        print(k, v)

if __name__ == "__main__":
    evaluate("../checkpoints/world_model.pt")

