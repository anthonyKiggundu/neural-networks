import torch

class WorldModelValidator:
    def __init__(self, model):
        self.model = model.eval()

        self.thresholds = {
            'max_load': 0.85,
            'min_signal': 0.30
        }

    @torch.no_grad()
    def run_counterfactual_rollout(self, S_t, E_t, horizon=20):
        states = []
        current = S_t

        for _ in range(horizon):
            current, _ = self.model.predict_step(current, E_t)
            states.append(current)

        rollout = torch.stack(states)
        return self.compute_metrics(rollout)

    def compute_metrics(self, rollout):
        peak_load = rollout[:,0].max()
        avg_signal = rollout[:,2].mean()

        overload = "HIGH" if peak_load > self.thresholds['max_load'] else "LOW"
        qos = "NEGATIVE" if avg_signal < self.thresholds['min_signal'] else "STABLE"

        return {
            "peak_load": peak_load.item(),
            "overload_risk": overload,
            "predicted_qos": qos
        }

