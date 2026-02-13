"""
LLM-based KPI decision agent for 6G network management.
"""

import torch
import logging

logging.getLogger("accelerate.big_modeling").setLevel(logging.ERROR)


class LLMKPIAgent:
    """An LLM agent for network KPI decision-making with contextual memory."""
    
    def __init__(self, name="Agent", drift_tag=None, tokenizer=None, model=None, device=None):
        """
        Initialize the LLM KPI Agent.
        
        Args:
            name (str): Name identifier for the agent
            drift_tag (str): Drift tag for inference specialization
            tokenizer: HuggingFace tokenizer instance
            model: HuggingFace model instance
            device (str): Device to run inference on
        """
        self.name = name
        self.drift_tag = drift_tag or "baseline"
        self.tokenizer = tokenizer
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.tokenizer is None or self.model is None:
            raise ValueError(f"{self.name}: Model and tokenizer must be provided.")

        print(f"{self.name} initialized on device {self.device}.")

    def infer(self, kpis):
        """
        Use the LLM to infer decisions based on KPIs.
        
        Args:
            kpis (dict): Network KPI data
            
        Returns:
            dict: Recommended decisions and insights
        """
        prompt = self._build_prompt(kpis)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7
                )

            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_text = decoded_output[len(prompt):].strip()
            
            return self._parse_decision(new_text)

        except Exception as e:
            print(f"{self.name}: Failed to generate inference: {e}")
            return {"decision": "error", "fraud_detected": False}

    def _build_prompt(self, kpis):
        """Build the inference prompt from KPI data."""
        return f"""
Device: {kpis['device']}
Timestamp: {kpis.name}
Location: (Latitude: {kpis['Latitude']}, Longitude: {kpis['Longitude']}, Altitude: {kpis['Altitude']})
Mobility:
  - Speed: {kpis['speed_kmh']} km/h
  - Traffic Jam Factor: {kpis['Traffic Jam Factor']}
Network KPIs:
  - Latency (ping_ms): {kpis['ping_ms']}
  - Jitter: {kpis['jitter']}
  - Datarate: {kpis['datarate']}
  - Target Datarate: {kpis['target_datarate']}
Signal Quality (PCell):
  - RSRP: {kpis['PCell_RSRP_1']} dBm
  - RSRQ: {kpis['PCell_RSRQ_1']} dB
  - SNR: {kpis['PCell_SNR_1']} dB
Resource Utilization:
  - Downlink Resource Blocks: {kpis['PCell_Downlink_Num_RBs']}
  - Uplink Resource Blocks: {kpis['PCell_Uplink_Num_RBs']}
Current Observations:
  - Reported Quality of Service (QoS): {kpis['measured_qos']}
  - Operator ID: {kpis['operator']}

Please provide:
1. Risk classification ("low", "moderate", "high", or "critical").
2. Evaluate the presence of fraud (True/False) and provide a rationale.
3. Suggestions for governance credits, penalties, or pricing adjustments.
4. Insights into optimal performance improvements.
5. Evaluate whether handover to a neighboring cell is advised.
6. Recommendations to prioritize specific traffic flows.
7. Based on my current trajectories, I will lose QoS in 120 seconds.
"""

    def _parse_decision(self, decision_text):
        """Extract structured data from decision text."""
        import re
        
        try:
            clean_text = decision_text.lower()
            fraud_detected = "true" in clean_text or "fraud" in clean_text
            
            risk_match = re.search(r"risk classification: (\w+)", clean_text)
            risk = risk_match.group(1).capitalize() if risk_match else "Moderate"
            
            decision_summary = f"Risk: {risk}. Fraud: {fraud_detected}. Insights: {decision_text[:50]}..."
            
            return {
                "decision": decision_summary,
                "risk_classification": risk,
                "fraud_detected": fraud_detected,
                "governance_adjustments": "Standard",
                "raw_text": decision_text
            }
        except Exception as e:
            return {
                "decision": f"Parsing Error: {str(e)}",
                "fraud_detected": False,
                "risk_classification": "Unknown",
                "governance_adjustments": "None"
            }
