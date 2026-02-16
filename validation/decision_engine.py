def decision_engine(metrics):
    if metrics['overload_risk'] == "HIGH":
        return "REJECT", "Predicted bottleneck > 85% load."

    if metrics['predicted_qos'] == "NEGATIVE":
        return "REJECT", "Signal quality drops below threshold."

    return "APPROVE", "Network remains stable."

