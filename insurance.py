import numpy as np
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


class ConservativePlanner(InsurableModel):
    def infer(self, kpis):
        return {
            "bt": kpis["confidence"] * 0.9,
            "latency": kpis["latency"] * 1.1
        }

class AggressivePlanner(InsurableModel):
    def infer(self, kpis):
        return {
            "bt": kpis["confidence"],
            "latency": kpis["latency"] * 0.8
        }

class InsurableModel:

    def __init__(self, profile):
        self.profile = profile

    def infer(self, kpis):
        return {
            "bt": confidence,
            "latency": lv,
            "intent_alignment": score
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
        
        # Risk Weighting Factors (The Gamma, Beta, Delta from your LaTeX)
        self.gamma = 25.0     # Epistemic Risk Weight (Uncertainty)
        self.beta = 8.0       # Environmental Risk Weight (Jitter)
        self.delta = 20.0     # Staleness Risk Weight (Latency Penalty)
        self.zeta_max = 0.35  # Max Governance Credit (35% discount)

    def generate_telemetry(self):
        """Generates realistic 6G telemetry: Jitter, True Confidence, and Reported Confidence."""
        t = np.arange(self.steps)
        
        # Phase 1: Nominal V2X Operation (0-50)
        # Phase 2: Adversarial Attack / Sensor Drift (50-100)
        # Phase 3: Malicious Metadata Reporting (100-150)
        
        jitter = 0.05 + 0.1 * np.random.rand(self.steps)
        bt_true = 0.95 * np.ones(self.steps)
        
        # Phase 2: High Jitter + Low Confidence
        jitter[50:100] += 0.55
        bt_true[50:100] -= 0.6
        
        # Phase 3: Malicious Reporting (Low true confidence, but Agent claims high confidence)
        jitter[100:150] += 0.45
        bt_true[100:150] -= 0.7
        
        bt_reported = bt_true.copy()
        bt_reported[100:150] = 0.9  # The "Malicious Lie"
        
        return jitter, bt_true, bt_reported

    def calculate_premium(self, bt_rep, bt_true, jitter, lv, step):
        """Calculates the dynamic premium with Governance Credit revocation logic."""
        # 1. Calculate Risk Components
        epistemic_risk = self.gamma * (1 - bt_rep)
        env_risk = self.beta * jitter
        staleness_risk = self.delta * max(0, lv - self.dt_req)
        
        total_risk_premium = self.p_base + epistemic_risk + env_risk + staleness_risk
        
        # 2. Evaluate Governance Credit (zeta)
        # Credit is revoked if Reported Confidence is high but Environment Jitter is also high
        # (Indicates the Agent is ignoring the safety context / out-of-distribution state)
        zeta = self.zeta_max
        if step > 100 and jitter > 0.4 and bt_rep > 0.8:
            zeta = 0.0  # FRAUD DETECTED: Revoke Safety Dividend
            
        return total_risk_premium * (1 - zeta)

    def evaluate(self):
        jitter, bt_true, bt_rep = self.generate_telemetry()
        p_history = []
        lv_history = []
        
        for i in range(self.steps):
            # Agent logic: Increase reasoning depth (Lv) as confidence (Bt) drops
            # But Agent is aware of the 'delta' penalty, so it stays near 'dt_req'
            target_lv = 0.4 + (1 - bt_rep[i]) * 0.9
            lv_opt = min(target_lv, self.dt_req + 0.2) # Adaptive Reasoning
            
            p = self.calculate_premium(bt_rep[i], bt_true[i], jitter[i], lv_opt, i)
            p_history.append(p)
            lv_history.append(lv_opt)
            
        return jitter, bt_true, bt_rep, lv_history, p_history

def apply_coverage_limits(premium, risk, coverage):
    uncovered = False

    if risk["epistemic"] > coverage.max_epistemic_risk:
        uncovered = True
    if risk["network"] > coverage.max_network_risk:
        uncovered = True
    if risk["staleness"] > coverage.max_staleness_risk:
        uncovered = True

    if uncovered:
        premium *= coverage.uncovered_penalty  # Self-insurance kicks in

    return min(premium, coverage.premium_cap)


def model_risk_multiplier(profile):
    return 1 + profile.fraud_penalty


def underwrite_model(profile, telemetry):
    if np.mean(telemetry["bt_true"]) < profile.min_confidence:
        return False, "Rejected: Low epistemic reliability"

    if np.max(telemetry["jitter"]) > profile.max_jitter:
        return False, "Rejected: Excessive jitter sensitivity"

    if np.mean(telemetry["lv"]) > profile.max_latency_violation:
        return False, "Rejected: SLA instability"

    return True, "Approved"

def premium_kernel(self, risk, bt_rep):
    scaling = (
        1
        + self.gamma * risk["epistemic"]
        + self.beta * risk["network"]
        + self.delta * risk["staleness"]
    )
    confidence_credit = np.exp(-bt_rep)
    return self.p_base * scaling * confidence_credit


def governance_credit(self, risk, bt_rep, step):
    if risk["fraud"] and step > 100:
        return 0.0
    return self.zeta_max



for model in models:
    metadata = model.infer(kpis)
    risk = risk_engine(metadata, kpis)
    premium = premium_engine(risk, metadata)

# --- Run and Visualize ---
evaluator = Agentic6GInsuranceEval()
jitter, b_true, b_rep, lv, premiums = evaluator.evaluate()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Top Plot: Premium Dynamics
ax1.plot(premiums, color='crimson', linewidth=2.5, label='Dynamic Premium ($P_t$)')
ax1.set_ylabel('Insurance Cost ($)')
ax1.set_title('Integrated 6G AI Insurance Evaluation: Adversarial & Governance Stress-Test')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')

# Bottom Plot: Metadata & Forensics
ax2.plot(b_true, color='blue', linestyle='--', alpha=0.5, label='True Internal Confidence')
ax2.plot(b_rep, color='green', linewidth=2, label='Reported Metadata ($B_T$)')
ax2.plot(lv, color='orange', label='Verification Latency ($L_v$)')
ax2.axhline(y=1.0, color='black', linestyle=':', label='6G SLA ($\Delta t_{req}$)')
ax2.set_ylabel('Metadata Value')
ax2.set_xlabel('Decision Epoch (6G Transmission Time Intervals)')
ax2.fill_between(range(50, 100), 0, 1.2, color='gray', alpha=0.1, label='Adversarial Interference')
ax2.fill_between(range(100, 150), 0, 1.2, color='red', alpha=0.05, label='Metadata Manipulation Zone')
ax2.legend(loc='lower left')

plt.tight_layout()
plt.show()
