"""
Helper utility functions for GIRAF.
"""


def print_decision(step, agent_name, decision):
    """
    Print a structured decision report.
    
    Args:
        step (int): Current simulation step
        agent_name (str): Name of the agent
        decision (dict): Decision dictionary
    """
    print(f"Step {step} Decision ({agent_name}):")
    print(f"  - Risk Classification: {decision.get('risk_classification', 'N/A')}")
    print(f"  - Fraud Detected: {'Yes' if decision.get('fraud_detected', False) else 'No'}")
    print(f"  - Governance Adjustments: {decision.get('governance_adjustments', 'N/A')}")
    print("\n")


def initialize_telemetry():
    """
    Initialize telemetry logging dictionary.
    
    Returns:
        dict: Empty telemetry dictionary
    """
    return {
        'time': [], 'risk': [], 'lv': [], 'gap': [],
        'bt_true': [], 'bt_rep': [], 'jitter': [],
        'congestion': [], 'epi_risk': [], 'staleness_risk': []
    }


def record_telemetry(log, **kwargs):
    """
    Record telemetry data.
    
    Args:
        log (dict): Telemetry log dictionary
        **kwargs: Key-value pairs to record
    """
    for key, value in kwargs.items():
        if key in log:
            log[key].append(value)
