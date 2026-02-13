"""
Risk calculation and quantification for GIRAF.
"""

import math


def calculate_risk_factors(metadata, evaluator, step, fraud_detected, behavior_flagged, threshold=45.0):
    """
    Calculate the Aggregate Risk Index (R_t) and generate a GaC Mitigation Signal.
    
    Args:
        metadata (dict): Real-time telemetry
        evaluator (dict): Weights and requirements
        step (int): Current decision epoch
        fraud_detected (bool): Fraud signal from LLM
        behavior_flagged (bool): Behavioral anomaly signal
        threshold (float): Risk threshold for mitigation
        
    Returns:
        dict: Risk components and mitigation signal
    """
    # 1. Epistemic Risk (Uncertainty)
    bt_confidence = metadata.get("bt_true", 1.0)
    r_epi = evaluator.get("gamma", 25.0) * (1 - bt_confidence)
    
    # 2. Environmental Risk
    omega = metadata.get("Traffic Jam Factor", 0) / 10.0
    r_env = evaluator.get("beta", 8.0) * (omega ** 2)
    
    # 3. Staleness Risk
    lv = float(metadata.get('ping_ms', 0))
    if lv > 5000:
        lv = 5000  # Cap extreme values
    
    dt_req = evaluator.get("dt_req", 25.0)
    r_stal = evaluator.get("delta", 20.0) * max(0, lv - dt_req)
    
    # 4. Aggregate Base Risk
    aggregate_risk = r_epi + r_env + r_stal
    
    # 5. Strategic Mitigation Factor
    coverage = metadata.get("constraint_coverage", 0.8)
    zeta = 0.15 * math.log(1 + coverage)
    final_risk = aggregate_risk * (1 - zeta)
    
    # 6. Adversarial Penalty
    if fraud_detected:
        final_risk += 30.0
    if behavior_flagged:
        final_risk += 15.0
    
    final_risk = max(final_risk, 0)
    
    # 7. Mitigation Signal
    mitigation_signal = 1 if final_risk > threshold else 0
    
    # 8. Trust Score
    trust_score = max(0, 100 - (final_risk * 1.5))
    
    return {
        "aggregate_risk": final_risk,
        "epistemic_component": r_epi,
        "staleness_component": r_stal,
        "environmental_component": r_env,
        "mitigation_signal": mitigation_signal,
        "trust_score": trust_score,
        "lv": lv,
        "smt_depth": 8  # Default SMT depth
    }
