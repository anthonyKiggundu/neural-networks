#!pip install -q --upgrade transformers accelerate peft bitsandbytes
#import os
#os.kill(os.getpid(), 9)

import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import requests
import pandas as pd
from sklearn.model_selection import train_test_split  # Add this
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, default_data_collator, EarlyStoppingCallback, pipeline, DataCollatorWithPadding, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset, DatasetDict  # Simplify creating and loading datasets
from torchvision import models

from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW

import logging
logging.getLogger("accelerate.big_modeling").setLevel(logging.ERROR)


EPOCH_DURATION_MS = 100  # Example: Each decision epoch is 100ms

# --- KPI Stream Simulation ---
def stream_kpis_from_dataset(df, feed_interval=1):
    """
    Stream KPI data dynamically row by row from a dataframe.
    :param df: Pandas DataFrame containing the KPI data.
    :param feed_interval: Time interval (in seconds) between each step.
    """
    for _, row in df.iterrows():
        yield row.to_dict()
        time.sleep(feed_interval)


class LLMKPIAgent:
    def __init__(self, name="Agent", drift_tag=None, tokenizer=None, model=None, device=None):
        """
        An LLM agent for network KPI decision-making with contextual memory.
        Args:
            model_name: Pretrained model identifier or path.
            drift_tag: Drift tag (e.g., "baseline", "low") for inference specialization.
            name (str): Name identifier for the agent.
            token (str): Hugging Face authentication token for gated/private models.
        """
        #self.model_name = model_name
        self.name = name
        self.drift_tag = drift_tag or "baseline"
        self.tokenizer = tokenizer
        self.model = model
        #self.token = token
        #self.history = []  # Maintain a decision history for contextual reasoning

        # Determine device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.tokenizer is None or self.model is None:
            raise ValueError(f"{self.name}: Model and tokenizer must be provided.")

        print(f"{self.name} initialized on device {self.device}.")


    def infer(self, kpis):
        """
        Use the LLM to infer decisions based on KPIs and past context.
        Args:
            kpis: Dictionary containing network KPI data (a single row from the dataframe).

        Returns:
            dict: Recommended decisions and insights based on KPIs.
        """
        # Build a prompt using the selected KPI data
        prompt = f"""
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
        4. Insights into optimal performance improvements:
           a. Signal quality optimization.
           b. Energy efficiency improvements.
           c. Congestion reduction strategies.
        5. Evaluate whether handover to a neighboring cell is advised given the current network and mobility behavior.
        6. Recommendations to prioritize specific traffic flows based on distance and congestion index.
        7. Based on my current trajectories, I will lose QoS in 120 seconds.
        """

        try:
            # Tokenize the input prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(**inputs,
                                          max_new_tokens=150,
                                          pad_token_id=self.tokenizer.eos_token_id, # Add this line
                                          eos_token_id=self.tokenizer.eos_token_id,
                                          do_sample=True,
                                          temperature=0.7
                                      )

            # Decode response
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the prompt from the output to get only the agent's "new" text
            new_text = decoded_output[len(prompt):].strip()

            # Parse into the DICTIONARY format the main loop expects
            return self._parse_decision(new_text)

        except Exception as e:
            print(f"{self.name}: Failed to generate inference: {e}")
            # Return a valid dictionary even on failure to prevent "string indices" error
            return {"decision": "error", "fraud_detected": False}


    def _parse_decision(self, decision_text):
        """
        Extracts structured data.
        Crucially adds the 'decision' key your loop is looking for.
        """
        try:
            # Clean up the text for searching
            clean_text = decision_text.lower()

            # 1. Check for Fraud (looks for 'true' or the word 'fraud' in the text)
            fraud_detected = "true" in clean_text or "fraud" in clean_text

            # 2. Extract Risk Classification
            import re # Import re module
            risk_match = re.search(r"risk classification: (\w+)", clean_text)
            risk = risk_match.group(1).capitalize() if risk_match else "Moderate"

            # 3. Create the 'decision' summary key that your loop uses:
            # fraud_detected = "fraud" in decision["decision"].lower()
            decision_summary = f"Risk: {risk}. Fraud: {fraud_detected}. Insights: {decision_text[:50]}..."

            return {
                "decision": decision_summary, # Key required by your loop
                "risk_classification": risk,
                "fraud_detected": fraud_detected,
                "raw_text": decision_text
            }
        except Exception as e:
            return {
                "decision": f"Parsing Error: {str(e)}",
                "fraud_detected": False
            }


def print_decision(step, agent_name, decision):
    """
    Print a well-structured decision report for the given step.
    Args:
        step (int): The current step in the simulation.
        agent_name (str): Name of the agent making the decision.
        decision (dict): The structured decision returned by the agent.
    """
    print(f"Step {step} Decision ({agent_name}):")
    print(f"  - Risk Classification: {decision['risk_classification']}")
    print(f"  - Fraud Detected: {'Yes' if decision['fraud_detected'] else 'No'}")
    print(f"  - Governance Adjustments: {decision['governance_adjustments']}")
    print("\n")


def calculate_risk_factors(
    metadata, evaluator, step, fraud_detected, behavior_flagged, threshold=45.0
):
    """
    Calculates the Aggregate Risk Index (R_t) and generates a GaC Mitigation Signal.

    Args:
        metadata (dict): Real-time telemetry (ping, jitter, bt_true, etc.)
        evaluator (dict): weights (gamma, beta, delta) and requirements (dt_req)
        step (int): Current decision epoch
        fraud_detected (bool): Signal from the LLM agent
        behavior_flagged (bool): Behavioral anomaly signal
        threshold (float): The Risk Index value at which mitigation is triggered

    Returns:
        dict: Risk components, aggregate index, and the binary mitigation signal.
    """
    # 1. Epistemic Risk (Uncertainty): gamma * (1 - B_T)
    # Reflects the gap between agent confidence and ground truth
    bt_confidence = metadata.get("bt_true", 1.0)
    r_epi = evaluator.get("gamma", 25.0) * (1 - bt_confidence)

    # 2. Environmental Risk: beta * omega^2
    # Reflects network volatility (Traffic Jam Factor / Jitter)
    omega = metadata.get("Traffic Jam Factor", 0) / 10.0
    r_env = evaluator.get("beta", 8.0) * (omega**2)

    # 3. Staleness Risk: delta * (Lv - dt_req)+
    # Reflects the "Verification-Staleness Trade-off"
    lv = metadata.get("ping_ms", 0)
    dt_req = evaluator.get("dt_req", 25.0)
    r_stal = evaluator.get("delta", 20.0) * max(0, lv - dt_req)

    # 4. Aggregate Base Risk
    aggregate_risk = r_epi + r_env + r_stal

    # 5. Strategic Mitigation Factor (zeta)
    # Based on constraint coverage (how much of the state space is formally guarded)
    coverage = metadata.get("constraint_coverage", 0.8)
    zeta = 0.15 * math.log(1 + coverage)

    final_risk = aggregate_risk * (1 - zeta)

    # 6. Adversarial Penalty
    if fraud_detected:
        final_risk += 30.0  # Heavy weight for intentional manipulation
    if behavior_flagged:
        final_risk += 15.0

    # Sanity check: ensure ping isn't causing a mathematical explosion
    lv = float(metadata.get('ping_ms', 0))
    # If ping is > 5000, it's likely an error or a different unit
    if lv > 5000: lv = 5000

    r_stal = evaluator.get("delta", 20.0) * max(0, lv - evaluator.get("dt_req", 25.0))

    # Ensure risk is non-negative
    final_risk = max(final_risk, 0)

    # 7. GENERATE MITIGATION SIGNAL (Governance-as-Code Trigger)
    # This signal tells the 6G control plane to intervene
    mitigation_signal = 1 if final_risk > threshold else 0

    # Calculate a "System Trust Score" [0-100] as an inverse of risk
    trust_score = max(0, 100 - (final_risk * 1.5))

    return {
        "aggregate_risk": final_risk,
        "epistemic_component": r_epi,
        "staleness_component": r_stal,
        "environmental_component": r_env,
        "mitigation_signal": mitigation_signal,
        "trust_score": trust_score
    }



def generate_reliability_diagram(y_true, prob_pred, n_bins=10):
    # 1. Convert to NumPy arrays
    y_true = np.array(y_true)
    prob_pred = np.array(prob_pred)

    # 2. Sync lengths to avoid dimension errors
    min_len = min(len(y_true), len(prob_pred))
    y_true = y_true[:min_len]
    prob_pred = prob_pred[:min_len]

    # 3. FIX: Binarize y_true
    # If y_true contains values like 0.733, we turn it into 1 (Success) 
    # if it's above a certain performance threshold, otherwise 0.
    # Alternatively, if y_true was meant to be "did it pass the trust check":
    y_true_binary = (y_true > 0.5).astype(int) 

    # 4. Clean NaNs
    mask = ~np.isnan(y_true_binary) & ~np.isnan(prob_pred)
    
    try:
        bin_true, bin_pred = calibration_curve(y_true_binary[mask], prob_pred[mask], n_bins=n_bins)
        
        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        plt.plot(bin_pred, bin_true, marker='s', color='green', label='GIRAF Agent')
        plt.fill_between(bin_pred, bin_pred, bin_true, color='pink', alpha=0.2, label='Calibration Error')
        plt.xlabel(r'Reported Confidence ($B_R$)')
        plt.ylabel('Empirical Accuracy')
        plt.title('Reliability Diagram (6G-V2X Confidence)')
        plt.legend()
        plt.show()
    except ValueError as e:
        print(f"Calibration Error: {e}. Check if y_true_binary contains both 0 and 1.")


def extended_visualize_results(time_span, aggregate_risks, epistemic_risks, staleness_risks,
                               congestion_index, jitter, bt_true, bt_reported,
                               ping_violations, jitter_violations, fraud_detected):

    # Ensure all arrays match the first dimension to prevent ValueErrors
    min_len = min(len(time_span), len(aggregate_risks), len(epistemic_risks), len(staleness_risks))
    t = time_span[:min_len]

    # 6 subplots to visualize the full GIRAF context
    fig, axes = plt.subplots(6, 1, figsize=(12, 22), sharex=True)

    # 0. Aggregate Risk
    #axes[0].plot(time_span, aggregate_risks, label='Aggregate Risk ($R_t$)', color='black', linewidth=2)
    #axes[0].axhline(y=45.0, color='r', linestyle='--', label='Mitigation Threshold')
    #axes[0].set_title("GIRAF Governance: Risk Indexing")
    #axes[0].legend()

    # --- SUBPLOT 1: Aggregate Risk (Fixed Scaling) ---
    axes[0].plot(t, aggregate_risks[:min_len], color='black', label=r'Aggregate Risk ($R_t$)')
    axes[0].axhline(y=45, color='red', linestyle='--', label='Mitigation Threshold')
    # Use symlog to handle values from 0 to 60,000 without losing the baseline
    axes[0].set_yscale('symlog', linthresh=100)
    axes[0].set_ylabel(r'Risk Index ($R_t$) [SymLog]')
    axes[0].set_title("GIRAF Governance: Risk Indexing")
    axes[0].legend(loc='upper right')

    # 1. Network Environment
    #axes[1].plot(time_span, jitter, label='Jitter (ms)', color='orange')
    #axes[1].plot(time_span, congestion_index, label='Traffic Jam Factor', color='brown', alpha=0.6)
    #axes[1].set_title("Environmental Context")
    #axes[1].legend()

    # --- Environmental Context Fix ---
    ax_env = axes[1]
    ax_env.plot(t, congestion_index, label='Traffic Jam Factor', color='brown', alpha=0.6)
    ax_env.set_ylabel('Congestion Index', color='brown')

    ax_jitter = ax_env.twinx()
    ax_jitter.plot(t, jitter, label='Jitter (ms)', color='orange', linewidth=1.5)
    ax_jitter.set_ylabel('Jitter (ms)', color='orange')
    ax_jitter.set_ylim(0, max(jitter) * 1.2 if any(jitter) else 10)

    # 2. Risk Components
    #axes[2].plot(time_span, epistemic_risks, label='Epistemic (LLM)', color='blue')
    #axes[2].plot(time_span, staleness_risks, label='Staleness (Latency)', color='green')
    #axes[2].set_title("Risk Component Decomposition")
    #axes[2].legend()

    # --- SUBPLOT 3: Risk Component Decomposition (Fixed Scaling) ---
    # We use dual Y-axes here because Epistemic (~0.5) and Staleness (~60,000) cannot coexist on one linear scale
    ax3 = axes[2]
    ax3.plot(t, epistemic_risks[:min_len], label='Epistemic (LLM)', color='blue', linewidth=1.5)
    ax3.set_ylabel('Epistemic Risk', color='blue')

    ax3_twin = ax3.twinx()
    ax3_twin.plot(t, staleness_risks[:min_len], label='Staleness (Latency)', color='green', alpha=0.5)
    ax3_twin.set_ylabel('Staleness Risk', color='green')
    ax3_twin.set_yscale('log') # Staleness is the high-magnitude culprit

    ax3.set_title("Risk Component Decomposition (Dual-Scaled)")
    # Combine legends
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='upper left')

    # 3. Binary Incident Flags
    axes[3].step(time_span, ping_violations, label='Ping Violation', where='post', color='red', alpha=0.5)
    axes[3].step(time_span, fraud_detected, label='Fraud Detected', where='post', color='darkred', linewidth=2)
    axes[3].set_title("Binary Incident Flags")
    axes[3].legend()

    # 4. Confidence Alignment
    # --- 4. Agentic Confidence Alignment (FIXED) ---
    # Ensure all arrays match the first dimension of time_span to prevent ValueErrors
    min_len_n = min(len(time_span), len(bt_true), len(bt_reported))
    t_plot = time_span[:min_len_n]

    axes[4].plot(t_plot, bt_true[:min_len_n], 'k--', label=r'Ground Truth ($B_T$)', alpha=0.8)
    axes[4].plot(t_plot, bt_reported[:min_len_n], 'g-', label=r'Reported ($B_R$)')

    # Set explicit limits to fix the Y-axis scale issue
    axes[4].set_ylim(-0.05, 1.05)
    axes[4].set_ylabel("Confidence [0,1]")
    axes[4].set_title("Agentic Confidence Alignment")
    axes[4].legend(loc='upper right')

    #axes[4].plot(time_span, bt_true, 'k--', label='Ground Truth ($B_T$)', alpha=0.8)
    #axes[4].plot(time_span, bt_reported, 'g-', label='Reported ($B_R$)')
    #axes[4].set_title("Agentic Confidence Alignment")
    #axes[4].legend()

    # 5. The Confidence Gap (Dissonance)
    # gap = np.abs(np.array(bt_true) - np.array(bt_reported))
    #gap = np.abs(bt_true[:min_len_n] - bt_reported[:min_len_n])
    #axes[5].fill_between(time_span, gap, color='purple', alpha=0.3, label='Confidence Gap ($r_{epi}$)')
    #axes[5].set_title("Epistemic Dissonance (Risk Magnitude)")
    #axes[5].set_xlabel(r"Time Span ($t+\tau$)") # Explicitly using t+tau
    #axes[5].set_ylabel("Error")
    #axes[5].legend()

    # Ensure inputs are arrays for math operations
    bt_t = np.array(bt_true)
    bt_r = np.array(bt_reported)

    # Sync lengths
    min_len_n = min(len(time_span), len(bt_t), len(bt_r))
    t = np.array(time_span)[:min_len_n]

    # Calculate Gap safely
    gap = np.abs(bt_t[:min_len_n] - bt_r[:min_len_n])

    # Plotting Subplot 6
    axes[5].fill_between(t, gap, color='purple', alpha=0.3, label=r'Confidence Gap ($r_{epi}$)')
    axes[5].axhline(y=0.15, color='red', linestyle='--', label=r'Trust Boundary ($\Omega$)')
    axes[5].set_ylim(0, 1.0) # Gap is between 0 and 1
    axes[5].set_ylabel('Error')
    axes[5].legend()

    plt.tight_layout()
    # plt.savefig("extended_simulation_results.png")
    # Use bbox_inches='tight' to ensure labels aren't cut off in the PDF
    plt.savefig("GIRAF_Simulation_Results.pdf", format='pdf', bbox_inches='tight', dpi=300)
    print("Plot successfully saved as GIRAF_Simulation_Results.pdf")
    plt.show()


def old_extended_visualize_results(time_seconds, aggregate_risks, epistemic_risks, staleness_risks, # time_steps,
                               congestion_index, jitter, bt_true, bt_reported,
                               ping_violations, jitter_violations, fraud_detected):

    # 1. FIX THE DATA: Log-scale for the 'Exploding' Risk
    # This ensures the Epistemic risk (small) and Staleness risk (huge) are both visible
    log_aggregate = np.log10(np.array(aggregate_risks) + 1)
    log_epistemic = np.log10(np.array(epistemic_risks) + 1)
    log_staleness = np.log10(np.array(staleness_risks) + 1)

    fig, axs = plt.subplots(5, 1, figsize=(14, 25), sharex=True) # Changed to 5 subplots
    plt.subplots_adjust(hspace=0.2)

    # --- PLOT 1: THE MOBILITY RISK PROFILE (Log-Scale) ---
    # This shows the novelty of the Aggregate Risk Index
    ##axs[0].plot(time_steps, log_aggregate, label="Total Risk ($R_t$)", color="red", linewidth=2)

    axs[0].plot(time_seconds, log_aggregate, label="Total Risk ($R_t$)", color="red", linewidth=1)
    axs[0].set_xlabel("Time (seconds)") # Label as seconds
    axs[0].fill_between(time_seconds, log_aggregate, color="red", alpha=0.1)
    axs[0].axhline(y=np.log10(45), color='black', linestyle='--', label="GaC Threshold")
    # axs[0].set_ylabel("Risk Score ($\log_{10}$)")
    axs[0].set_ylabel(r"Risk Score ($\log_{10}$)")
    axs[0].set_title("GIRAF: Dynamic Risk Profile During Mobility Traversal")
    axs[0].legend(loc="upper right")

    # --- PLOT 2: RISK DECOMPOSITION (Stacked Log) ---
    # NOW you will see the Epistemic risk because of the log scale
    ##axs[1].fill_between(time_steps, 0, log_epistemic, label="Epistemic (Uncertainty)", color="purple", alpha=0.6)
    ##axs[1].fill_between(time_steps, log_epistemic, log_epistemic + log_staleness,
    # label="Staleness (Latency)", color="orange", alpha=0.6)

    axs[1].fill_between(time_seconds, 0, log_epistemic, label="Epistemic (Uncertainty)", color="purple", alpha=0.6)
    axs[1].fill_between(time_seconds, log_epistemic, log_epistemic + log_staleness,
                        label="Staleness (Latency)", color="orange", alpha=0.6)
    axs[1].set_ylabel("Risk Contribution")
    axs[1].set_title("Verification-Staleness Trade-off Analysis")
    axs[1].legend(loc="upper left")

    # --- PLOT 3: ENVIRONMENTAL TELEMETRY (Dual Axis) ---
    # This explains 'Why' the risk is changing as the device moves
    ax3_twin = axs[2].twinx()
    ##lns1 = axs[2].plot(time_steps, jitter, label="Jitter (ms)", color="blue", alpha=0.6)
    ##lns2 = ax3_twin.plot(time_steps, congestion_index, label="Congestion", color="brown", linewidth=2)

    lns1 = axs[2].plot(time_seconds, jitter, label="Jitter (ms)", color="blue", alpha=0.6)
    lns2 = ax3_twin.plot(time_seconds, congestion_index, label="Congestion", color="brown", linewidth=2)
    axs[2].set_ylabel("Network Flux")
    ax3_twin.set_ylabel("Traffic Factor")
    axs[2].set_title("Environmental Context: Congestion vs. Jitter")
    # Merge legends
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    axs[2].legend(lns, labs, loc="upper right")

    # --- PLOT 4: TRUST & FRAUD INDICATORS ---
    # Replaces the boring Plot 4 with a "Trust Score" + Fraud Spikes
    # Trust Score is 100 - Normalized Risk
    trust_score = 100 * (1 - (np.array(aggregate_risks) / (max(aggregate_risks) + 1)))
    ##axs[3].plot(time_steps, trust_score, label="System Trust Score", color="green")
    ##axs[3].step(time_steps, np.array(fraud_detected)*50, label="Fraud Detected", color="red", where='post')

    axs[3].plot(time_seconds, trust_score, label="System Trust Score", color="green")
    axs[3].step(time_seconds, np.array(fraud_detected)*50, label="Fraud Detected", color="red", where='post')
    axs[3].set_ylim(0, 110)
    axs[3].set_ylabel("Score / Indicator")
    axs[3].set_xlabel("Decision Epoch ($t$)")
    axs[3].set_title("Governance Plane: Real-time Trust Modulation")
    axs[3].legend(loc="upper right")

    # NEW SUBPLOT: Confidence Gap (Dissonance)
    # This shows the "Gap" we are talking about in the paper
    gap = np.abs(np.array(bt_true) - np.array(bt_reported))
    axs[4].plot(time_seconds, gap, label='Epistemic Gap ($r_{epi}$)', color='purple')
    axs[4].axhline(y=0.15, color='r', linestyle='--', label='Trust Threshold')
    axs[4].set_title("Real-time Confidence Dissonance (GIRAF Metric)")
    axs[4].set_ylabel("Error Magnitude")
    axs[4].set_xlabel("Decision Epoch ($t$)") # Add x-label for the new subplot
    axs[4].legend()

    plt.tight_layout()
    plt.savefig("extended_simulation_results.png")
    plt.show()


# --- Fine-Tuning ---
def fine_tune_model(train_data, base_model_name, tokenizer, val_data=None, save_dir="./finetuned_model", max_steps=100):
    """
    Fine-tune a base model with a training dataset using LoRA (Low-Rank Adaptation).
    Args:
        train_data: Pandas DataFrame containing the training data.
        base_model_name: Pretrained model name (e.g., 'EleutherAI/gpt-neo-1.3B' or 'NetGPT').
        tokenizer: Tokenizer for the model.
        val_data: Pandas DataFrame containing the validation data (optional).
        save_dir: Directory to save the fine-tuned model.
        max_steps: Maximum number of training steps.
    """

    accelerator = Accelerator()
    print("Preprocessing the training dataset for fine-tuning...")

    # Convert to HF datasets
    hf_train_dataset = Dataset.from_pandas(train_data)
    hf_val_dataset = Dataset.from_pandas(val_data) if val_data is not None else None

    # Explicitly add a pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # batched preprocessing
    def preprocess_function(batch):
        prompts = batch["kpi_input"]
        targets = batch["kpi_description"]

        full_texts = [p + "\n" + t + tokenizer.eos_token for p, t in zip(prompts, targets)]

        tokenized = tokenizer(
            full_texts,
            truncation=True,
            max_length=512,
            padding="max_length",
        )

        labels = []
        for i, text in enumerate(full_texts):
            input_ids = tokenized["input_ids"][i]
            # Create labels: start as a copy of input_ids
            label = list(input_ids)

            # Find where the target starts to mask the prompt
            # A more robust way: tokenize prompt only to find length
            tokenized_prompt = tokenizer(prompts[i] + "\n", truncation=True, max_length=512)
            prompt_len = len(tokenized_prompt["input_ids"])

            # Mask prompt tokens with -100
            for j in range(len(label)):
                if j < prompt_len:
                    label[j] = -100
                # Also mask padding tokens
                if input_ids[j] == tokenizer.pad_token_id:
                    label[j] = -100
            labels.append(label)

        tokenized["labels"] = labels
        return tokenized

    processed_train_dataset = hf_train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=hf_train_dataset.column_names,
    )

    processed_val_dataset = None
    if hf_val_dataset is not None:
        processed_val_dataset = hf_val_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=hf_val_dataset.column_names,
        )

    print("Setting up data loaders...")

    # Use default data collator for consistency
    data_collator = default_data_collator

    train_loader = DataLoader(
        processed_train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=data_collator,
    )

    val_loader = None
    if processed_val_dataset is not None:
        val_loader = DataLoader(
            processed_val_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=data_collator,
        )

    print("Initializing base model...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    # resize embeddings
    base_model.resize_token_embeddings(len(tokenizer))

    # Ensure model knows pad token
    base_model.config.pad_token_id = tokenizer.pad_token_id

    print("Adding LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        inference_mode=False,
    )

    model = get_peft_model(base_model, lora_config)
    model.train()

    # Sanity check
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters found.")
    print(f"Trainable parameters: {len(trainable)}")

    optimizer = AdamW(model.parameters(), lr=2e-4)

    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=2,  # Stop after 2 epochs of no improvement.
        early_stopping_threshold=0.002  # Stop when loss improvement is below this value.
    )

    # Adjusted epoch loop to use validation loss tracking
    best_loss = float("inf")
    early_stopping_triggered = False

    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    step_count = 0
    print("Starting fine-tuning...")
    for epoch in range(3):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            outputs = model(**batch)
            loss = outputs.loss

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            step_count += 1
            if step_count % 10 == 0:  # Print updates every 10 steps
                print(f"Epoch {epoch+1}, Step {step_count}, Loss {loss.item():.4f}")

            # Break loop after `max_steps` is reached
            if step_count >= max_steps:
                print(f"Reached max training steps: {max_steps}")
                break

        if step_count >= max_steps:
            break  # Exit outer loop once max_steps is reached

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f} Average Training Loss: {avg_train_loss:.4f}")

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(**batch)
                    val_loss += outputs.loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")

            # Check for early stopping
            if avg_val_loss < best_loss - 1e-4:  # Improvement threshold
                best_loss = avg_val_loss
                print(f"Validation loss improved to {avg_val_loss:.4f}")
            else:
                print("Early stopping triggered. No improvement in validation loss.")
                early_stopping_triggered = True

        # Break loop if early stopping has been triggered.
        if early_stopping_triggered:
            break

    save_path = f"{save_dir}/{base_model_name.replace('/', '_')}"
    print(f"Saving fine-tuned model to {save_path}...")
    accelerator.wait_for_everyone()
    model.save_pretrained(save_path, save_function=accelerator.save)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}.")

    return save_path


def get_dynamic_threshold(kpis, base_req=40.0, safety_floor=10.0):
    """
    Computes Latency SLA based on Speed and Congestion.
    """
    v = kpis.get("speed_kmh", 0)                # From your column list
    c = kpis.get("Traffic Jam Factor", 0)       # From your column list
    v_ref = 120.0                               # Reference speed in km/h

    # Formula: Threshold tightens as speed or congestion increases
    dynamic_limit = base_req * (1 - (c/10)) * math.exp(-v / v_ref) + safety_floor
    return max(dynamic_limit, safety_floor)

def get_dynamic_jitter_threshold(kpis, nominal_jitter=15.0, floor=2.0):
    """
    Computes Jitter SLA based on SNR and Congestion.
    """
    snr = kpis.get("PCell_SNR_1", 20)           # From your column list
    c = kpis.get("Traffic Jam Factor", 0)       # From your column list
    snr_max = 30.0

    # Jitter tolerance drops if signal is noisy or road is congested
    dynamic_jitter_limit = nominal_jitter * (snr / snr_max) * math.exp(-0.5 * (c/10)) + floor
    return max(dynamic_jitter_limit, floor)


def custom_compute_metrics_function(eval_pred):
    """Custom callback for evaluation metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}



def calculate_confidence_gap(model, tokenizer, prompt, ground_truth_val):

    # Determine the model's current device (cuda:0)
    device = next(model.parameters()).device

    # Get logits for the prediction
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # CRITICAL: Move all input tensors to the model's device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :] # Last token logits
            probs = torch.softmax(logits, dim=-1)
    except Exception as e:
        print(f"Inference error: {e}")
        return 1.0, 0.5 # Return high risk/low confidence on failure to avoid nan

    # Extract the scalar prediction (e.g., predicted latency)
    # This is a simplified version of your 'infer' function
    prediction = model.generate(**inputs, max_new_tokens=1)

    # Calculate Epistemic Uncertainty (r_epi)
    # r_epi = |Predicted_Confidence - Reality|
    # For LLMs, we often use the max probability as 'Self-Reported Confidence'
    reported_conf = torch.max(probs).item()

    # Calculate if the prediction matches ground truth (0 or 1)
    # Placeholder for actual prediction_matches_truth logic
    # For now, let's assume `ground_truth_val` is somehow used to determine accuracy
    # and that the model's generated output is 'prediction'.
    # Since the task does not define `ground_truth_val` for this `calculate_confidence_gap` function,
    # I will assume that if `ground_truth_val` is a string, we check if the prediction contains it.
    # This is a placeholder and should be refined with actual ground truth data if available.
    decoded_prediction = tokenizer.decode(prediction[0], skip_special_tokens=True)
    prediction_matches_truth = ground_truth_val in decoded_prediction if isinstance(ground_truth_val, str) else True # simplified
    accuracy = 1.0 if prediction_matches_truth else 0.0

    r_epi = abs(reported_conf - accuracy)
    return r_epi, reported_conf


def old_get_calibration_curve(confidences, labels, n_bins=10):
    bin_boundaries = np.linspace(0.5, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    accuracies = []
    conf_means = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if np.sum(in_bin) > 0:
            accuracies.append(np.mean(labels[in_bin]))
            conf_means.append(np.mean(confidences[in_bin]))

    return np.array(conf_means), np.array(accuracies)


def get_calibration_curve(labels, predictions, n_bins=10):
    # Convert to numpy arrays for indexing
    labels = np.array(labels)
    predictions = np.array(predictions)

    # FIX: Force alignment by truncating to the shortest array
    min_len = min(len(labels), len(predictions))
    labels = labels[:min_len]
    predictions = predictions[:min_len]

    # Proceed with binning logic
    bin_centers = np.linspace(0, 1, n_bins + 1)
    accuracies = []
    confidences = []

    for i in range(n_bins):
        # Create the boolean mask based on the aligned length
        in_bin = (predictions >= bin_centers[i]) & (predictions < bin_centers[i+1])

        if np.any(in_bin):
            # Now labels[in_bin] will never throw an IndexError
            accuracies.append(np.mean(labels[in_bin]))
            confidences.append(np.mean(predictions[in_bin]))

    return np.array(accuracies), np.array(confidences)


def initialize_telemetry():
    """Returns a dictionary of empty lists for all 6G/GIRAF metrics."""
    return {
        'time': [], 'risk': [], 'lv': [], 'gap': [],
        'bt_true': [], 'bt_rep': [], 'jitter': [],
        'congestion': [], 'epi_risk': [], 'staleness_risk': []
    }

def record_telemetry(log, **kwargs):
    """Appends data to telemetry logs, ensuring synchronization."""
    for key, value in kwargs.items():
        if key in log:
            log[key].append(value)



from sklearn.calibration import calibration_curve

def plot_reliability_diagram(y_true, prob_pred, n_bins=10):
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    prob_pred = np.array(prob_pred)

    # Sync dimensions (Fixes the 300 vs 450 error)
    min_len = min(len(y_true), len(prob_pred))
    y_true = y_true[:min_len]
    prob_pred = prob_pred[:min_len]

    # FORCE BINARY: Ensure y_true is strictly 0 or 1
    # Any value > 0.5 becomes 1, else 0.
    y_true_binary = (y_true > 0.5).astype(int)

    # Check if we have at least one of each class to avoid sklearn errors
    if len(np.unique(y_true_binary)) < 2:
        print("Warning: Data contains only one class. Reliability diagram may be skewed.")

    try:
        bin_true, bin_pred = calibration_curve(y_true_binary, prob_pred, n_bins=n_bins)

        plt.figure(figsize=(7, 7))
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal Calibration')
        plt.plot(bin_pred, bin_true, marker='s', color='green', label='GIRAF Agent')

        plt.fill_between(bin_pred, bin_pred, bin_true, color='pink', alpha=0.2, label='Calibration Error')

        plt.xlabel(r'Reported Confidence ($B_R$)')
        plt.ylabel('Empirical Success Rate')
        plt.title('Reliability Diagram: SLA & Trust Alignment')
        plt.legend()
        plt.grid(alpha=0.2)
        plt.show()

    except Exception as e:
        print(f"Plotting failed: {e}")


def old_plot_reliability_diagram(y_true, prob_pred, n_bins=10):
    
    """
    Fixes IndexError by aligning labels and predictions before binning.
    """
    # 1. Convert to NumPy arrays
    y_true = np.array(y_true)
    prob_pred = np.array(prob_pred)

    # 2. Force alignment: truncate to the shortest common length
    # This prevents the "300 vs 450" dimension mismatch
    min_len = min(len(y_true), len(prob_pred))
    y_true = y_true[:min_len]
    prob_pred = prob_pred[:min_len]

    # 3. Handle NaNs that might have been produced by CUDA/inference errors
    mask = ~np.isnan(y_true) & ~np.isnan(prob_pred)
    y_true, prob_pred = y_true[mask], prob_pred[mask]

    if len(y_true) == 0:
        print("Warning: No valid data for Reliability Diagram.")
        return

    # 4. Use sklearn's optimized calibration curve (replaces manual binning)
    # This avoids manual indexing errors like "boolean index did not match"
    bin_true, bin_pred = calibration_curve(y_true, prob_pred, n_bins=n_bins)

    # Plotting code
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.plot(bin_pred, bin_true, marker='s', color='green', label='GIRAF Agent')
    plt.fill_between(bin_pred, bin_pred, bin_true, color='pink', alpha=0.2, label='Calibration Error')

    plt.xlabel(r'Reported Confidence ($B_R$)')
    plt.ylabel('Empirical Accuracy')
    plt.title('Reliability Diagram (6G-V2X Confidence)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()



def plot_risk_distribution_by_traffic_jam_factor(congestion_data, epistemic_risks, staleness_risks):
    # Align and Categorize
    min_len = min(len(congestion_data), len(epistemic_risks))
    tj = np.array(congestion_data[:min_len])
    er = np.array(epistemic_risks[:min_len])
    sr = np.array(staleness_risks[:min_len])

    # Binning logic
    low = tj <= 3
    med = (tj > 3) & (tj <= 7)
    high = tj > 7

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Epistemic (Linear Scale)
    ax1.bar(['Low', 'Med', 'High'], 
            [np.mean(er[low]) if any(low) else 0, 
             np.mean(er[med]) if any(med) else 0, 
             np.mean(er[high]) if any(high) else 0], color='purple')
    ax1.set_title("Epistemic Risk by Congestion")

    # Plot Staleness (Log Scale - Crucial for high values)
    ax2.bar(['Low', 'Med', 'High'], 
            [np.mean(sr[low]) if any(low) else 0, 
             np.mean(sr[med]) if any(med) else 0, 
             np.mean(sr[high]) if any(high) else 0], color='gold')
    ax2.set_yscale('log')
    ax2.set_title("Staleness Risk (Log Scale)")
    
    plt.savefig("risk_distribution.pdf", format='pdf', bbox_inches='tight')


def old_plot_risk_distribution_by_traffic_jam_factor(
    congestion_data, epistemic_risks, staleness_risks
):
    """
    Visualizes the distribution (median, 90th percentile, and maximum) of
    epistemic_risks and staleness_risks as a function of 'Traffic Jam Factor'
    (binned into low, medium, high categories).
    """
    if not congestion_data or not epistemic_risks or not staleness_risks:
        print("Warning: No data to plot for risk distribution by Traffic Jam Factor.")
        return

    # Create a DataFrame for easier processing
    plot_df = pd.DataFrame({
        'Traffic Jam Factor': congestion_data,
        'Epistemic Risk': epistemic_risks,
        'Staleness Risk': staleness_risks
    })

    # Define bins for Traffic Jam Factor
    bins = [0, 3, 7, 10] # Assuming TJF ranges from 0 to 10
    labels = ['Low Congestion', 'Medium Congestion', 'High Congestion']
    plot_df['Congestion Category'] = pd.cut(
        plot_df['Traffic Jam Factor'], bins=bins, labels=labels, right=False,
        include_lowest=True
    )

    # Group by congestion category and calculate statistics
    grouped_data = plot_df.groupby('Congestion Category').agg(
        median_epi=('Epistemic Risk', 'median'),
        p90_epi=('Epistemic Risk', lambda x: x.quantile(0.9)),
        max_epi=('Epistemic Risk', 'max'),
        median_stal=('Staleness Risk', 'median'),
        p90_stal=('Staleness Risk', lambda x: x.quantile(0.9)),
        max_stal=('Staleness Risk', 'max')
    ).reindex(labels) # Ensure categories are in order

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Epistemic Risk Plot
    grouped_data[['median_epi', 'p90_epi', 'max_epi']].plot(
        kind='bar', ax=axes[0], colormap='viridis', alpha=0.8
    )
    axes[0].set_title('Epistemic Risk Distribution by Congestion')
    axes[0].set_xlabel('Congestion Category')
    axes[0].set_ylabel('Epistemic Risk Value')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend(['Median', '90th Percentile', 'Maximum'])

    # Staleness Risk Plot
    grouped_data[['median_stal', 'p90_stal', 'max_stal']].plot(
        kind='bar', ax=axes[1], colormap='plasma', alpha=0.8
    )
    axes[1].set_title('Staleness Risk Distribution by Congestion')
    axes[1].set_xlabel('Congestion Category')
    axes[1].set_ylabel('Staleness Risk Value') # Shared Y-axis might make this redundant if sharey=True
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(['Median', '90th Percentile', 'Maximum'])

    plt.suptitle('Risk Distribution by Traffic Jam Factor (SMT Depth Proxy)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig('risk_distribution_by_traffic_jam_factor.png')
    plt.show()
    print("Risk Distribution by Traffic Jam Factor plot saved as risk_distribution_by_traffic_jam_factor.png")


import seaborn as sns

def plot_verification_staleness_dist(lv_history, smt_depths, sla_deadline=1.0):
    plt.figure(figsize=(10, 6))

    # Ensure data is numeric and clean
    lv = np.array(lv_history)
    phi = np.array(smt_depths)

    plt.scatter(phi, lv, alpha=0.5, color='blue', label='Empirical $L_v$')
    plt.axhline(y=sla_deadline, color='darkred', linestyle='--',
                label=fr'SLA Deadline ($\Delta t_{{req}}$ = {sla_deadline}ms)')

    # Crucial: Use symlog to handle the 1ms vs 60,000ms range
    plt.yscale('symlog', linthresh=2.0)

    plt.xlabel(r"SMT Verification Depth ($\Phi$)", fontsize=12)
    plt.ylabel(r"Latency ($L_v$) [ms]", fontsize=12)
    plt.title(r"Empirical Verification Latency ($L_v$) vs. SMT Depth")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig("verified_latency_vs_SMTdepth.png")
    plt.show()

def iold_plot_verification_staleness_dist(lv_history, smt_depth_history, sla_deadline=60):
    """
    Uses real simulation telemetry to show the distribution of
    Verification Latency (Lv) across different SMT Depths.
    """
    # Create a DataFrame from the actual simulation logs
    df = pd.DataFrame({
        'Latency': lv_history,
        'SMT_Depth': smt_depth_history
    })

    plt.figure(figsize=(10, 6))

    # 1. Boxplot of actual data
    sns.boxplot(x='SMT_Depth', y='Latency', data=df, color='skyblue',
                width=0.5, showfliers=False, medianprops={'color': 'black', 'linewidth': 2})

    # 2. Overlay 90th percentile and Max for each unique depth found in the logs
    unique_depths = sorted(df['SMT_Depth'].unique())
    for i, depth in enumerate(unique_depths):
        subset = df[df['SMT_Depth'] == depth]['Latency']
        p90 = subset.quantile(0.9)
        pmax = subset.max()

        plt.plot(i, p90, 'r^', markersize=8, label='90th Percentile' if i == 0 else "")
        plt.plot(i, pmax, 'rx', markersize=8, label='Maximum' if i == 0 else "")


    plt.xlabel(r"SMT Verification Depth ($\Phi$)", fontsize=12)

    # 3. SLA Deadline
    plt.axhline(y=sla_deadline, color='darkred', linestyle='--',
            label=fr'SLA Deadline ($\Delta t_{{req}}$ = {sla_deadline}ms)')

    plt.title("Empirical Verification Latency ($L_v$) vs. SMT Depth", fontsize=14)
    plt.xlabel(r"SMT Verification Depth ($\Phi$)", fontsize=12)
    plt.ylabel("Latency ($L_v$) [ms]", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle=':', alpha=0.7)

    # Shade the 'Staleness Risk Zone'
    plt.fill_between([-0.5, len(unique_depths)-0.5], sla_deadline, plt.ylim()[1],
                     color='red', alpha=0.05)

    plt.tight_layout()
    plt.savefig("real_verification_staleness_dist.png", dpi=300)
    plt.show()


# --- Main Pipeline ---
def main():
    # Initialize the evaluator, agents, and contract
    evaluator = {
        "steps": 150,
        "dt_req": 1.0,  # Latency Deadline (ms)
        "p_base": 10.0,  # Baseline Premium ($)
        "gamma": 25.0,   # Epistemic Risk Weight (Uncertainty)
        "beta": 8.0,     # Environmental Risk Weight (Jitter)
        "delta": 20.0,   # Staleness Risk Weight (Latency Penalty)
    }
    contract = {
        "premium_cap": 50.0,  # Maximum Premium Cap
        "uncovered_penalty": 2.0  # Penalty factor for exceeding premium cap
    }

    # Define total_epochs and time_seconds here, after evaluator is defined
    total_epochs = evaluator["steps"]
    time_seconds = np.arange(total_epochs) * (EPOCH_DURATION_MS / 1000.0)

    # --- ADD THESE TO YOUR HISTORY LISTS (Line ~165) ---
    bt_reported_history = []  # To store reported_conf
    accuracy_history = []     # To store binary success (1 if accurate, 0 if not)

    # Hugging Face authentication token
    huggingface_token = "hf_tokem"

    # Load the dataset
    data = pd.read_parquet("cellular_dataframe.parquet")
    data["Step"] = range(len(data))  # Add Step column if missingthe

    # Ensure critical columns exist for the risk model
    if "bt_true" not in data.columns: data["bt_true"] = 0.95
    if "constraint_coverage" not in data.columns: data["constraint_coverage"] = 0.8

    # Add a column in the dataset to act as input for the LLM
    data['kpi_input'] = (
        "Device: " + data['device'].astype(str) +
        "\nTimestamp: " + data.index.astype(str) +
        "\nLocation: " +
        "(Latitude: " + data['Latitude'].astype(str) +
        ", Longitude: " + data['Longitude'].astype(str) +
        ", Altitude: " + data['Altitude'].astype(str) + ")" +
        "\nMobility:" +
        "\n  - Speed: " + data['speed_kmh'].astype(str) + " km/h" +
        "\n  - Traffic Jam Factor: " + data['Traffic Jam Factor'].astype(str) +
        "\nNetwork KPIs:" +
        "\n  - Latency (ping_ms): " + data['ping_ms'].astype(str) +
        "\n  - Jitter: " + data['jitter'].astype(str) +
        "\n  - Datarate: " + data['datarate'].astype(str) +
        "\n  - Target Datarate: " + data['target_datarate'].astype(str) +
        "\nSignal Quality (PCell):" +
        "\n  - RSRP: " + data['PCell_RSRP_1'].astype(str) + " dBm" +
        "\n  - RSRQ: " + data['PCell_RSRQ_1'].astype(str) + " dB" +
        "\n  - SNR: " + data['PCell_SNR_1'].astype(str) + " dB" +
        "\nResource Utilization:" +
        "\n  - Downlink Resource Blocks: " + data['PCell_Downlink_Num_RBs'].astype(str) +
        "\n  - Uplink Resource Blocks: " + data['PCell_Uplink_Num_RBs'].astype(str)
    )

    # Add dummy column for kpi_description (target labels)
    data['kpi_description'] = "This should describe the response for the given KPIs."

    # Split the dataset
    train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    print(f"Data split into training ({len(train_data)} rows), validation ({len(val_data)} rows), "
          f"and testing ({len(test_data)} rows).")


    # Calculate thresholds from a known 'clean' portion of your dataset
    # Use np.nanpercentile to skip over missing values in the dataframe
    SLA_PING_THRESHOLD = np.nanpercentile(test_data['ping_ms'], 95)
    SLA_JITTER_THRESHOLD = np.nanpercentile(test_data['jitter'], 95)

    # Safety check: if the column is entirely empty, set a hardcoded standard
    if np.isnan(SLA_PING_THRESHOLD): SLA_PING_THRESHOLD = 30.0
    if np.isnan(SLA_JITTER_THRESHOLD): SLA_JITTER_THRESHOLD = 10.0

    print(f"Dynamic Thresholds Set: Ping={SLA_PING_THRESHOLD:.2f}ms, Jitter={SLA_JITTER_THRESHOLD:.2f}ms")

    # Check for GPU availability, else fallback to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Preprocess and fine-tune the model
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M") # EleutherAI/gpt-neo-1.3B") #
    model_name = fine_tune_model(train_data, "EleutherAI/gpt-neo-125M", tokenizer, val_data) # EleutherAI/gpt-neo-125M

    # The original base model name (not your local folder)
    base_model_name = "EleutherAI/gpt-neo-125M"
    # The local path where your adapter_config.json sits
    adapter_path = "./finetuned_model/EleutherAI_gpt-neo-125M/"

    try:
        print("Loading base model...")
        # Configure 8-bit quantization correctly
        # quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # 1. Load the skeleton (Base Model)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto"
            # quantization_config=quantization_config # load_in_8bit=True
        )

        # 2. Load the Tokenizer from your local folder (to get your special tokens)
        tokenizer = AutoTokenizer.from_pretrained(model_name) #adapter_path)

        # CRITICAL STEP: Resize the base model FIRST.
        # This expands the matrix to 50258 so the adapter's weights fit.
        if len(tokenizer) > base_model.config.vocab_size:
            print(f"Pre-resizing base model: {base_model.config.vocab_size} -> {len(tokenizer)}")
            base_model.resize_token_embeddings(len(tokenizer))
        # 3. Wrap the base model with your fine-tuned adapter
        # print(f"Applying adapter from {adapter_path}...")

        model = PeftModel.from_pretrained(base_model, model_name)

        # Fix the warning: Explicitly set pad_token to eos_token
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        # 4. Handle the embedding resize for your added tokens
        if len(tokenizer) > model.config.vocab_size:
            print(f"Resizing embeddings for {len(tokenizer)} tokens")
            model.resize_token_embeddings(len(tokenizer))


        # Performance tweaks
        model.gradient_checkpointing_enable()

        print("LoRA Adapter and Base Model loaded successfully!")

    except Exception as e:
        print(f"Failed to load: {e}")
        # Initialize model to None or raise a more specific error
        model = None # Ensure 'model' is bound even on failure


    # Initialize LLM agents with different drift tags
    agents = [
        LLMKPIAgent(
            #model_name=model_name,  # "meta-llama/Llama-2-7b-hf",  # Replace with a valid Hugging Face model name
            drift_tag=str(i),
            name=f"KPI Agent {i}",
            tokenizer=tokenizer,
            model=model,
            device="cuda" if torch.cuda.is_available() else "cpu"
            #token=huggingface_token
        ) for i in range(1) #2
    ]

    # Store the historical premium values for visualization
    aggregate_risk_history, staleness_risks, epistemic_risks = [], [], []
    jitter_history, bt_true, bt_reported, congestion_index = [], [], [], []
    congestion_index, accuracy_history = [], []
    ping_violations_history, jitter_violations_history = [], []
    fraud_events, lv_history, smt_depth_history = [], [], [] # Binary (1 = Fraud, 0 = No Fraud)
    #congestion_history, ping_violations, fraud_events = [], [], []

    # Store risk factors for fraud and behavioral analysis
    fraud_flags = []
    behavior_flags = []
    signal_quality = {"RSRP": [], "RSRQ": [], "SNR": []}
    sla_violations = {"ping_ms": 0, "jitter": 0, "datarate": 0, "target_datarate": 0}

    # Initialization for Statistics
    total_decisions = 0
    mitigation_events = 0
    fraud_detected_count = 0
    high_risk_low_jitter = 0 # Cases where risk was high but network was "fine"

    bt_true_history = []
    bt_reported_history = []
    accuracy_history = []

    # Initialize these as empty lists before the loop
    all_labels = []
    all_preds = []

    # Constants for the mathematical definitions
    LAMBDA_SENSITIVITY = 0.25
    TRUST_THRESHOLD = 0.15

    # Stream KPI data from the test dataset and iterate
    for step, kpis in enumerate(stream_kpis_from_dataset(test_data, feed_interval=0.2)): #stream_kpis_real_time(150, feed_interval=0.1)): # NOTE: Changed back to test_data for continuity

        if step >= evaluator["steps"]: break

        fraud_detected = False
        behavior_flagged = False

        # We derive the 'True Confidence' from the jitter.
        # As jitter increases, the 'Ground Truth' certainty decreases.
        base_confidence = 0.95
        snr_val = kpis.get('snr', 20)
        jitter_val = kpis.get('jitter', 0.5)
        jitter_penalty = kpis.get('jitter', 0) / 100.0
        # current_bt_true = max(0.1, base_confidence - jitter_penalty)

        # Dynamic BT calculation based on telemetry
        # Formula: BT = exp(-lambda * (sigma_jitter / SNR))
        current_bt_true = np.exp(-LAMBDA_SENSITIVITY * (jitter_val / max(1, snr_val)))

        # 1. STORE TRUE VALUE
        bt_true_history.append(current_bt_true)

        # Update the kpis dictionary so the risk function sees the new value
        kpis['bt_true'] = current_bt_true

        prompt = kpis.get('kpi_input', "")

        #if llm_prompt == "Missing Data":
        #    print(f"Warning: Step {step} is missing 'kpi_input' column!")
        #    continue

        # 1. FIX MAPPING: Use 'ping_ms' instead of 'latency'
        # Use .get() to avoid KeyErrors if a row is malformed
        current_ping = kpis.get("ping_ms", 0)
        current_jitter = kpis.get("jitter", 0)

        jitter_history.append(current_jitter)
        # bt_true/reported are likely not in your CSV; we default them to 1.0 (100% confidence)
        #bt_true.append(kpis.get("bt_true", 1.0))
        #bt_reported.append(kpis.get("bt_reported", 1.0))
        congestion_index.append(kpis.get("Traffic Jam Factor", 0))

        # 2. CALCULATE DYNAMIC SLA LIMITS
        sla_p_limit = get_dynamic_threshold(kpis)
        sla_j_limit = get_dynamic_jitter_threshold(kpis)

        step_confidences = []
        step_preds = []

        # 3. AGENT INFERENCE
        # Passing the pre-formatted 'kpi_input' string to the agent
        for agent in agents:
            try:
                # 1. Run the Governor function we defined earlier
                # Note: TODO:: You'll need to define 'y_true' based on your kpis['kpi_description']
                # Placeholder for actual prediction_matches_truth logic
                # Use kpis['kpi_description'] as a placeholder for ground truth value needed by calculate_confidence_gap
                prediction_matches_truth = True  # Define here for calculate_confidence_gap
                try:
                    r_epi, reported_conf = calculate_confidence_gap(model, tokenizer, prompt, kpis['kpi_description'])
                    step_confidences.append(reported_conf)
                except Exception as e:
                    print(f"Inference crash at step {step}: {e}")
                    step_confidences.append(0.5) # Safe fallback

                # 3. SYNCHRONIZATION: Record exactly ONE value per step
                avg_reported_conf = np.mean(step_confidences)

                bt_true_history.append(current_bt_true)
                bt_reported_history.append(avg_reported_conf)
    
                # 4. Success Criteria (For Reliability Diagram)
                # Success = High Temporal Reliability (Latency) AND High Epistemic Reliability (Dissonance)
                is_timely = lv_history[-1] <= SLA_DEADLINE if lv_history else True
                is_reliable = np.abs(current_bt_true - avg_reported_conf) < TRUST_THRESHOLD
    
                actual_success = 1 if (is_timely and is_reliable) else 0
                accuracy_history.append(actual_success)

                # 2. Use this in your real-time risk calculation
                # Override the kpis dict so calculate_risk_factors sees the actual LLM uncertainty
                kpis['bt_true'] = current_bt_true
                kpis['bt_reported'] = reported_conf
                kpis['r_epi'] = r_epi

                decision = agent.infer(prompt) #kpis["kpi_input"])
                # If decision is a dictionary:
                if isinstance(decision, dict):
                    decision_text = str(decision.get("decision", "")).lower()
                # If decision is just a string:
                else:
                    decision_text = str(decision).lower()

                fraud_detected = fraud_detected or "fraud" in decision_text #decision["decision"].lower()
                behavior_flagged = behavior_flagged or "flagged" in decision_text

                risk_data = calculate_risk_factors(
                    kpis, evaluator, step, fraud_detected, behavior_flagged, threshold=45.0
                )

                step_preds.append(reported_conf)

                # --- C. SYNCHRONIZED APPENDING (Prevents ValueError) ---
                # Every list gets exactly one append per step
                # bt_true_history.append(current_bt_true)
                # bt_reported_history.append(reported_conf)
                jitter_history.append(jitter_val)
                congestion_index.append(kpis.get("Traffic Jam Factor", 0))

                aggregate_risk_history.append(risk_data["aggregate_risk"])
                epistemic_risks.append(risk_data["epistemic_component"])
                staleness_risks.append(risk_data["staleness_component"])

                # Track Latency for the Distribution Plot
                lv_history.append(risk_data.get("lv", 0))
                smt_depth_history.append(risk_data.get("smt_depth", 8)) # Default to 8 if not dynamic

                ping_violations_history.append(1 if kpis.get("ping_ms", 0) > SLA_PING_THRESHOLD else 0)
                fraud_events.append(1 if fraud_detected else 0)

                # 3. Log for the final Reliability Diagram
                # bt_reported_history.append(reported_conf) # Already appended above
                # accuracy_history.append(1 if r_epi < 0.15 else 0) # Already appended above

            except Exception as e:
                print(f"Inference error: {e}")
                # Ensure bt_reported_history and accuracy_history are kept in sync even on inference error
                bt_reported_history.append(0.5) # Default/unknown confidence
                accuracy_history.append(0)    # Assume inaccurate on error

        
        # 1. Define what "Success" means for this step
        # Option A: Success is based on the Epistemic Risk being low (no Dissonance)
        # Option B: Success is based on avoiding a Ping Violation
        # --- Define 6G-V2X Success Criteria ---
        # 1. Define Environmental Sensitivity
        #LAMBDA_SENSITIVITY = 0.25 

        # 2. Inside the loop, calculate BT dynamically
        # BT will vary between 0 and 1 based on network conditions
        #bt_true_val = np.exp(-LAMBDA_SENSITIVITY * (jitter / max(1, snr)))
        #bt_true_history.append(bt_true_val)

        # 3. SUCCESS DEFINITION (Must vary to avoid the "One Class" error)
        # A success is when the agent's reported confidence (BR) 
        # is within 15% of the objective ground truth (BT)
        #actual_success = 1 if np.abs(bt_true_val - reported_conf) < 0.15 else 0
        #all_labels.append(actual_success)

        #actual_success = 1 if r_epi < 0.15 else 0
        # FIX: Append only the average or the first agent's report to keep length at 150
        bt_reported_history.append(np.mean(step_confidences)) 

        # TO KEEP DIMENSIONS 1:1 WITH STEPS:
        # Append the average prediction of all agents for this specific step
        all_preds.append(np.mean(step_preds))

        # Append exactly one ground-truth label per step
        # all_labels.append(1 if actual_success else 0)

        # 4. TRACK VIOLATIONS (Binary spikes for the plot)
        ping_v = 1 if current_ping > sla_p_limit else 0
        jitter_v = 1 if current_jitter > sla_j_limit else 0

        ping_violations_history.append(ping_v)
        jitter_violations_history.append(jitter_v)
        fraud_events.append(1 if fraud_detected else 0)

        # D. RISK QUANTIFICATION
        risk_data = calculate_risk_factors(
            kpis, evaluator, step, fraud_detected, behavior_flagged, threshold=45.0
        )

        # After calling calculate_risk_factors
        total_decisions += 1
        if risk_data["mitigation_signal"] == 1:
            mitigation_events += 1
        if fraud_detected:
            fraud_detected_count += 1

        # Log the mitigation event to console for tracking
        if risk_data["mitigation_signal"]:
            print(f"⚠️ STEP {step}: GaC Mitigation Triggered! Risk Index: {risk_data['aggregate_risk']:.2f}")

        aggregate_risk_history.append(risk_data["aggregate_risk"])
        epistemic_risks.append(risk_data["epistemic_component"])
        staleness_risks.append(risk_data["staleness_component"])

        torch.cuda.empty_cache()  # Clear memory after each step


    # Call the visualization function after streaming is done
    # 6. FINAL VISUALIZATION
    # 1. First, call your extended visualizer
    #extended_visualize_results(
    #    time_seconds=time_seconds, # Use time_seconds instead of time_steps
    #    aggregate_risks=aggregate_risk_history,
    #    epistemic_risks=epistemic_risks,
    #    staleness_risks=staleness_risks,
    #    jitter=jitter_history,
    #    bt_true=bt_true,
    #    bt_reported=bt_reported_history, # Use the list we populated in the loop
    #    ping_violations=ping_violations_history,
    #    jitter_violations=jitter_violations_history,
    #    fraud_detected=fraud_events
    #)

    # --- AFTER THE LOOP ---
    tau = 0.2  # Match your feed_interval
    time_span = [step * tau for step in range(len(aggregate_risk_history))]

    # Call the updated visualizer
    extended_visualize_results(
        time_span=time_span, # Passing the calculated time span
        aggregate_risks=aggregate_risk_history,
        epistemic_risks=epistemic_risks,
        staleness_risks=staleness_risks,
        congestion_index=congestion_index,
        jitter=jitter_history,
        bt_true=bt_true_history,
        bt_reported=bt_reported_history,
        ping_violations=ping_violations_history,
        jitter_violations=jitter_violations_history,
        fraud_detected=fraud_events
    )

    # 2. GENERATE THE RELIABILITY DIAGRAM (Post-simulation check)
    print("Generating Reliability Diagram...")
    # Convert our tracked lists to numpy arrays
    # Ensure bt_reported_history and accuracy_history are not empty
    if bt_reported_history and accuracy_history:
        conf_arr = np.array(bt_reported_history)
        acc_arr = np.array(accuracy_history)

        # Calculate the points for the plot
        bins, accs = get_calibration_curve(conf_arr, acc_arr)

        # Call the plotting function
        plot_reliability_diagram(bins, accs)
    else:
        print("Not enough data to generate Reliability Diagram.")

    # 3. Generate the new risk distribution plot
    print("Generating Risk Distribution by Traffic Jam Factor plot...")
    plot_risk_distribution_by_traffic_jam_factor(
        congestion_data=congestion_index,
        epistemic_risks=epistemic_risks,
        staleness_risks=staleness_risks
    )


    # Convert lists to numpy arrays for calculation
    risk_arr = np.array(aggregate_risk_history)
    trust_arr = 100 * (1 - (risk_arr / (max(risk_arr) + 1e-6)))
    mitigation_arr = np.array([1 if r > 45.0 else 0 for r in aggregate_risk_history])

    # 1. Governance Effectiveness
    # mean_risk = np.mean(risk_arr)
    mean_risk = risk_array[~np.isnan(risk_arr)]
    peak_risk = np.max(risk_arr)
    mitigation_ratio = (np.sum(mitigation_arr) / len(mitigation_arr)) * 100

    # 2. Trust Modulation
    avg_trust = np.mean(trust_arr)
    min_trust = np.min(trust_arr)

    # Ensure they are the same length (handling multi-agent cases)
    # If 1 agent, lengths will be identical.
    bt_t_arr = np.array(bt_true_history)
    # Make sure bt_r_arr has the same length as bt_t_arr
    bt_r_arr = np.array(bt_reported_history[:len(bt_t_arr)])

    avg_gap = np.mean(np.abs(bt_t_arr - bt_r_arr))

    # 3. Confidence Gap (Epistemic Dissonance)
    # Identify steps where reported confidence was 1.0 but true confidence was low
    confidence_gap_events = sum(1 for bt_t, bt_r in zip(bt_true_history, bt_reported_history) if (bt_r - bt_t) > 0.1)

    # Peak Instability Index (Maximum Risk Spike)
    peak_instability = np.max(risk_arr) if len(risk_arr) > 0 else 0

    # Mean Aggregate Risk
    #mean_risk = np.mean(risk_arr) if len(risk_arr) > 0 else 0

    # Mitigation Trigger Rate (%)
    mitigation_threshold = 45.0
    mitigation_count = np.sum(risk_arr > mitigation_threshold)
    trigger_rate = (mitigation_count / len(risk_arr)) * 100 if len(risk_arr) > 0 else 0

    # Average System Trust Score
    # Defined as inverted normalized risk (0 to 100)
    trust_scores = 100 * (1 - (risk_arr / (peak_instability + 1e-6)))
    avg_trust = np.mean(trust_scores) if len(trust_scores) > 0 else 0

    print(f"--- Simulation Results for Table ---")
    print(f"Mean Confidence Gap (${{r_{{epi}}}}$): {avg_gap:.4f}") # <--- New Key Metric
    print(f"Confidence Gap Events: {confidence_gap_events}")
    #print(f"Mean Aggregate Risk: {mean_risk:.4f}")
    print(f"Mean Aggregate Risk: {np.mean(mean_risk):.4f}" if len(mean_risk) > 0 else "Mean Aggregate Risk: 0.0000")
    print(f"Peak Instability Index: {peak_instability:.2f}")
    print(f"Mitigation Trigger Rate: {trigger_rate:.1f}%")
    print(f"Average System Trust Score: {avg_trust:.2f}")
    #print(f"Epistemic Dissonance Events: {dissonance_events}")

    # --- CALCULATE GOVERNANCE METRICS ---
    gap_arr = np.abs(np.array(bt_true_history) - np.array(bt_reported_history))

    # 1. Total Accumulated Dissonance (AUC)
    # This uses the trapezoidal rule over the time span (t+tau)
    total_dissonance_auc = np.trapz(gap_arr, x=time_span)

    # 2. Untrusted Time Calculation
    trust_threshold = 0.15
    untrusted_mask = gap_arr > trust_threshold
    # Duration = Number of steps outside threshold * tau (time per step)
    untrusted_duration = np.sum(untrusted_mask) * tau
    total_duration = time_span[-1] if time_span else 0.0 # Handle empty time_span
    untrusted_ratio = (untrusted_duration / total_duration) * 100 if total_duration > 0 else 0.0

    print(f"\n--- Governance Impact Metrics ---")
    print(f"Total Accumulated Dissonance (AUC): {total_dissonance_auc:.4f} bits-sec")
    print(f"Total Untrusted Time: {untrusted_duration:.2f}s ({untrusted_ratio:.1f}% of mission)")

    # Execute
    # 2. Call the Staleness Distribution Plot
    # Ensure this function is defined as we discussed in the previous turn
    plot_verification_staleness_dist(lv_history, smt_depth_history, sla_deadline=evaluator["dt_req"])

    # Run for both
    #r_epi_base, conf_base = calculate_confidence_gap(pretrained_model, tk, k_t, y_true)
    #r_epi_tuned, conf_tuned = calculate_confidence_gap(finetuned_model, tk, k_t, y_true)

    # Simulated data reflecting 6G-V2X telemetry results
    #np.random.seed(42)
    #n_samples = 2000

    # 1. Pretrained Model: High confidence, but often wrong (Overconfident)
    #conf_pretrained = np.random.uniform(0.6, 1.0, n_samples)
    #acc_pretrained = (np.random.rand(n_samples) < (conf_pretrained * 0.72)).astype(int)

    # 2. Fine-tuned Agent: Confidence is well-aligned with task success
    #conf_tuned = np.random.uniform(0.5, 1.0, n_samples)
    #acc_tuned = (np.random.rand(n_samples) < (conf_tuned * 0.98)).astype(int)

    # Calculate curves
    #bins_pre, acc_pre = get_calibration_curve(conf_pretrained, acc_pretrained)
    #bins_ft, acc_ft = get_calibration_curve(conf_tuned, acc_tuned)

    # Plotting
    #plt.figure(figsize=(7, 6))
    #plt.plot([0.5, 1.0], [0.5, 1.0], '--', color='gray', label='Perfectly Calibrated')
    #plt.plot(bins_pre, acc_pre, 'o-', color='red', label='Pretrained LLM (Overconfident)')
    #plt.plot(bins_ft, acc_ft, 's-', color='green', label='Fine-tuned Agent (GIRAF Aligned)')

    # Visualizing the "Confidence Gap"
    #plt.fill_between(bins_pre, acc_pre, bins_pre, color='red', alpha=0.1, label='Confidence Gap')

    #plt.title('Reliability Diagram: Confidence Calibration in 6G Telemetry')
    #plt.xlabel('Self-Reported Confidence ($B_R$)')
    #plt.ylabel('Empirical Accuracy')
    #plt.legend()
    #plt.grid(True, linestyle=':', alpha=0.6)
    #plt.savefig('confidence_calibration_curve.png')

if __name__ == "__main__":
    main()
