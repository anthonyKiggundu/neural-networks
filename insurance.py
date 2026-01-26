mport numpy as np
import torch
import matplotlib.pyplot as plt

class Reinsurer:
    def __init__(self, attachment_point, coverage_share):
        self.attachment_point = attachment_point
        self.coverage_share = coverage_share  # e.g., 0.7 means 70%

    def absorb(self, premium):
        if premium > self.attachment_point:
            excess = premium - self.attachment_point
            return premium - excess * self.coverage_share
        return premium


class ConservativePlanner:
    def infer(self, kpis):
        return {
            "bt": kpis["confidence"] * 0.9,
            "latency": kpis["latency"] * 1.1
        }


class AggressivePlanner:
    def infer(self, kpis):
        return {
            "bt": kpis["confidence"],
            "latency": kpis["latency"] * 0.8
        }


class InsurableModel:
    def __init__(self, profile, pretrained_model=None):
        """
        :param profile: A profile object containing underwriting rules
        :param pretrained_model: A pretrained PyTorch model to use for inference, optional
        """
        self.profile = profile
        self.pretrained_model = pretrained_model  # Store the pretrained model if provided

    def infer(self, kpis):
        """
        Infer metadata using the pretrained model or fallback to default behavior.
        :param kpis: A dictionary of KPIs (e.g., confidence, latency)
        :return: Metadata extracted by the model
        """
        if self.pretrained_model:
            # Use the pretrained model for inference
            input_tensor = torch.tensor([kpis["confidence"], kpis["latency"]], dtype=torch.float32)
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

            self.pretrained_model.eval()
            with torch.no_grad():
                output = self.pretrained_model(input_tensor)

            # Assume output has two features: [bt, latency]
            return {
                "bt": float(output[0, 0]),  # Predicted confidence
                "latency": float(output[0, 1])  # Predicted latency
            }
        else:
            # Default inference if no pretrained model exists
            return {
                "bt": kpis["confidence"],
                "latency": kpis["latency"]
            }


class ModelUnderwritingProfile:
    def __init__(
        self,
        name,
        min_confidence,
        max_jitter,
        max_latency_violation,
        ood_tolerance,
        fraud_penalty
    ):
        self.name = name
        self.min_confidence = min_confidence
        self.max_jitter = max_jitter
        self.max_latency_violation = max_latency_violation
        self.ood_tolerance = ood_tolerance
        self.fraud_penalty = fraud_penalty


class CoverageContract:
    def __init__(
        self,
        max_epistemic_risk,
        max_network_risk,
        max_staleness_risk,
        premium_cap,
        uncovered_penalty=2.0
    ):
        self.max_epistemic_risk = max_epistemic_risk
        self.max_network_risk = max_network_risk
        self.max_staleness_risk = max_staleness_risk
        self.premium_cap = premium_cap
        self.uncovered_penalty = uncovered_penalty


class Agentic6GInsuranceEval:
    def __init__(self, steps=150):
        self.steps = steps
        self.dt_req = 1.0     # 6G Slice Latency Deadline (ms)
        self.p_base = 10.0    # Baseline premium ($)
        
        # Risk Weighting Factors
        self.gamma = 25.0     # Epistemic Risk Weight (Uncertainty)
        self.beta = 8.0       # Environmental Risk Weight (Jitter)
        self.delta = 20.0     # Staleness Risk Weight (Latency Penalty)
        self.zeta_max = 0.35  # Max Governance Credit (35% discount)


# --- Layered Pipeline ---

def stream_kpis(steps=150):
    t = np.arange(steps)
    jitter = 0.05 + 0.1 * np.random.rand(steps)
    bt_true = 0.95 * np.ones(steps)
    
    # Phase 2: High Jitter + Low Confidence
    jitter[50:100] += 0.55
    bt_true[50:100] -= 0.6

    # Phase 3: Malicious Reporting
    jitter[100:150] += 0.45
    bt_true[100:150] -= 0.7
    
    bt_reported = bt_true.copy()
    bt_reported[100:150] = 0.9  # The "Malicious Lie"
    
    return jitter, bt_true, bt_reported


def extract_metadata(jitter, bt_reported, bt_true):
    return {
        "bt_reported": bt_reported,
        "bt_true": bt_true,
        "jitter": jitter,
    }


def calculate_risk_factors(metadata, evaluator, step, planner):
    kpis = {"confidence": metadata["bt_reported"][step], "latency": metadata["jitter"][step]}
    inferred_metadata = planner.infer(kpis)

    epistemic_risk = evaluator.gamma * (1 - inferred_metadata["bt"])
    env_risk = evaluator.beta * metadata["jitter"][step]
    staleness_risk = evaluator.delta * max(0, evaluator.dt_req - inferred_metadata["latency"])
    return {"epistemic": epistemic_risk, "network": env_risk, "staleness": staleness_risk}


def premium_engine(risk_factors, evaluator, step, lv_opt, contract):
    total_risk_premium = (
        evaluator.p_base
        + risk_factors["epistemic"]
        + risk_factors["network"]
        + risk_factors["staleness"]
    )
    if total_risk_premium > contract.premium_cap:
        total_risk_premium *= contract.uncovered_penalty
    return total_risk_premium


def apply_governance_logic(premium, metadata, evaluator, step, reinsurer):
    zeta = evaluator.zeta_max
    if step > 100 and metadata["jitter"][step] > 0.4 and metadata["bt_reported"][step] > 0.8:
        zeta = 0.0  # Governance Credit revoked for fraud detection
    adjusted_premium = premium * (1 - zeta)
    return reinsurer.absorb(adjusted_premium), zeta


def visualize_results(premium_history, jitter, bt_true, bt_reported, lv_history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Premium Dynamics
    ax1.plot(premium_history, color="crimson", linewidth=2.5, label="Premium ($)")
    ax1.set_ylabel("Cost ($)")
    ax1.set_title("6G AI Insurance Evaluation")
    ax1.legend()
    ax1.grid(True)

    # Metadata
    ax2.plot(bt_true, label="True Confidence", linestyle="--", alpha=0.5)
    ax2.plot(bt_reported, label="Reported Confidence", linewidth=2, color="green")
    ax2.fill_between(range(len(bt_reported)), 0, 1.2, where=(np.array(jitter) > 0.4), color='gray', alpha=0.3, label="High Jitter Zone")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# --- Pretrained Model Integration Example ---
def load_pretrained_model():
    """
    Load a pretrained model (e.g., ResNet or an LSTM-based model).
    This is an example function. Replace model structure for your specific use case.
    """
    from torchvision import models
    from torch import nn

    # Example: Load a pretrained ResNet model (replace with a time-series-relevant model if necessary).
    model = models.resnet18(pretrained=True)

    # Modify the final fully-connected layer to output two features: [bt, latency].
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


# --- Main Pipeline ---
def main():
    # Initialization
    evaluator = Agentic6GInsuranceEval(steps=150)
    pretrained_model = load_pretrained_model()
    planner = ConservativePlanner()
    profile = ModelUnderwritingProfile("TestProfile", 0.8, 0.5, 1.2, 0.2, 1.1)
    contract = CoverageContract(0.8, 0.6, 0.4, 50.0)
    reinsurer = Reinsurer(30, 0.7)

    pretrained_model = load_pretrained_model()
    # Initialize InsurableModel with the pretrained model
    insurable_model = InsurableModel(profile, pretrained_model)

    # Generate simulated KPI data
    jitter, bt_true, bt_reported = stream_kpis(evaluator.steps)
    metadata = extract_metadata(jitter, bt_reported, bt_true)

    premium_history = []
    zeta_history = []
    lv_history = []

    for step in range(evaluator.steps):
        # Infer metadata using the pretrained model
        inferred_metadata = insurable_model.infer({
            "confidence": metadata["bt_reported"][step],
            "latency": metadata["jitter"][step]
        })

        # Agent decision logic
        target_lv = 0.4 + (1 - inferred_metadata["bt"]) * 0.9
        lv_opt = min(target_lv, evaluator.dt_req + 0.2)

        # Calculate risk factors and premiums
        risk = calculate_risk_factors(metadata, evaluator, step, planner)
        premium = premium_engine(risk, evaluator, step, lv_opt, contract)

        # Apply governance logic and reinsurer logic
        adjusted_premium, zeta = apply_governance_logic(premium, metadata, evaluator, step, reinsurer)

        # Record results
        premium_history.append(adjusted_premium)
        zeta_history.append(zeta)
        lv_history.append(lv_opt)

    # Visualization
    visualize_results(premium_history, jitter, bt_true, bt_reported, lv_history)


if __name__ == "__main__":
    main()
