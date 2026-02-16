import json
from datetime import datetime
from pathlib import Path

class ExperimentLogger:
    def __init__(self, name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(f"../runs/{name}_{timestamp}")
        self.dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.dir / "metrics.jsonl"

    def log(self, step, metrics):
        record = {"step": step, **metrics}
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def save_config(self, config):
        with open(self.dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def save_model(self, model):
        import torch
        torch.save(model.state_dict(), self.dir / "model.pt")

