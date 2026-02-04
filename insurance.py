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


# --- KPI Stream Simulation ---
import numpy as np


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
            inputs = self.tokenizer(prompt_text, return_tensors="pt", max_length=512, truncation=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=150)

            # Decode response
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the prompt from the output to get only the agent's "new" text
            new_text = decoded_output[len(prompt_text):].strip()

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


def calculate_risk_factors(metadata, evaluator, step, fraud_detected, behavior_flagged, threshold=45.0):
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
    r_env = evaluator.get("beta", 8.0) * (omega ** 2)

    # 3. Staleness Risk: delta * (Lv - dt_req)+
    # Reflects the "Verification-Staleness Trade-off" from your abstract
    lv = metadata.get('ping_ms', 0)
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
        final_risk += 30.0 # Heavy weight for intentional manipulation
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


def extended_visualize_results(time_steps, aggregate_risks, epistemic_risks, staleness_risks,
                               congestion_index, jitter, bt_true, bt_reported,
                               ping_violations, jitter_violations, fraud_detected):

    # 1. FIX THE DATA: Log-scale for the 'Exploding' Risk
    # This ensures the Epistemic risk (small) and Staleness risk (huge) are both visible
    log_aggregate = np.log10(np.array(aggregate_risks) + 1)
    log_epistemic = np.log10(np.array(epistemic_risks) + 1)
    log_staleness = np.log10(np.array(staleness_risks) + 1)

    fig, axs = plt.subplots(4, 1, figsize=(14, 20), sharex=True)
    plt.subplots_adjust(hspace=0.2)

    # --- PLOT 1: THE MOBILITY RISK PROFILE (Log-Scale) ---
    # This shows the novelty of the Aggregate Risk Index
    axs[0].plot(time_steps, log_aggregate, label="Total Risk ($R_t$)", color="red", linewidth=2)
    axs[0].fill_between(time_steps, log_aggregate, color="red", alpha=0.1)
    axs[0].axhline(y=np.log10(45), color='black', linestyle='--', label="GaC Threshold")
    # axs[0].set_ylabel("Risk Score ($\log_{10}$)")
    axs[0].set_ylabel(r"Risk Score ($\log_{10}$)")
    axs[0].set_title("GIRAF: Dynamic Risk Profile During Mobility Traversal")
    axs[0].legend(loc="upper right")

    # --- PLOT 2: RISK DECOMPOSITION (Stacked Log) ---
    # NOW you will see the Epistemic risk because of the log scale
    axs[1].fill_between(time_steps, 0, log_epistemic, label="Epistemic (Uncertainty)", color="purple", alpha=0.6)
    axs[1].fill_between(time_steps, log_epistemic, log_epistemic + log_staleness,
                        label="Staleness (Latency)", color="orange", alpha=0.6)
    axs[1].set_ylabel("Risk Contribution")
    axs[1].set_title("Verification-Staleness Trade-off Analysis")
    axs[1].legend(loc="upper left")

    # --- PLOT 3: ENVIRONMENTAL TELEMETRY (Dual Axis) ---
    # This explains 'Why' the risk is changing as the device moves
    ax3_twin = axs[2].twinx()
    lns1 = axs[2].plot(time_steps, jitter, label="Jitter (ms)", color="blue", alpha=0.6)
    lns2 = ax3_twin.plot(time_steps, congestion_index, label="Congestion", color="brown", linewidth=2)
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
    axs[3].plot(time_steps, trust_score, label="System Trust Score", color="green")
    axs[3].step(time_steps, np.array(fraud_detected)*50, label="Fraud Detected", color="red", where='post')
    axs[3].set_ylim(0, 110)
    axs[3].set_ylabel("Score / Indicator")
    axs[3].set_xlabel("Decision Epoch ($t$)")
    axs[3].set_title("Governance Plane: Real-time Trust Modulation")
    axs[3].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def old_extended_visualize_results(
    time_steps,
    aggregate_risks,
    epistemic_risks,
    staleness_risks,
    congestion_index,
    jitter,
    bt_true,
    bt_reported,
    ping_violations,
    jitter_violations,
    fraud_detected,
    risk_threshold=45.0  # New parameter for the GaC trigger line
):

    # Set up a figure with 6 subplots
    fig, axs = plt.subplots(6, 1, figsize=(16, 28), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    # --- Plot 1: Aggregate Risk Index (R_t) ---
    axs[0].plot(time_steps, aggregate_risks, label="Aggregate Risk Index ($R_t$)", color="red", linewidth=2.5)

    # Add the "Risk Threshold" dashed line
    axs[0].axhline(y=risk_threshold, color='black', linestyle='--', linewidth=2,
                  label=f"GaC Mitigation Threshold ({risk_threshold})")

    # Highlight areas where risk exceeds the threshold (Mitigation Zones)
    risk_array = np.array(aggregate_risks)
    axs[0].fill_between(time_steps, risk_array, risk_threshold,
                        where=(risk_array > risk_threshold),
                        color='orange', alpha=0.3, label="Active Mitigation Zone")

    axs[0].set_ylabel("Risk Score")
    axs[0].set_title("Scientific Aggregate Risk Index over 6G Control Loop")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, linestyle='--')

    # --- Plot 2: Component Risks (Stacked) ---
    axs[1].stackplot(time_steps, epistemic_risks, staleness_risks,
                     labels=['Epistemic Risk', 'Staleness Risk'],
                     colors=['purple', 'orange'], alpha=0.7)
    axs[1].set_ylabel("Risk Weight")
    axs[1].set_title("Decomposition of Latency-Staleness vs. Uncertainty")
    axs[1].legend(loc="upper left")
    axs[1].grid(True, alpha=0.3)

    # --- Plot 3: Network Integrity (Jitter & Congestion) ---
    ax3_twin = axs[2].twinx()
    axs[2].plot(time_steps, jitter, label="Network Jitter", color="blue", alpha=0.6)
    ax3_twin.plot(time_steps, congestion_index, label="Congestion Index", color="brown", linewidth=2)
    axs[2].set_ylabel("Jitter (ms)")
    ax3_twin.set_ylabel("Traffic Factor")
    axs[2].set_title("Environmental Flux: Jitter vs. Traffic Congestion")
    axs[2].legend(loc="upper left")
    ax3_twin.legend(loc="upper right")

    # --- Plot 4: Epistemic Truth vs. Reported Confidence (B_T) ---
    axs[3].plot(time_steps, bt_true, label="True Confidence ($B_T$)", color="green", linewidth=2)
    axs[3].plot(time_steps, bt_reported, label="Reported Confidence", color="black", linestyle="--", alpha=0.7)
    axs[3].set_ylabel("Confidence $[0,1]$")
    axs[3].set_title("Agent Epistemic Integrity: True vs. Reported States")
    axs[3].legend()
    axs[3].grid(True)

    # --- Plot 5: Binary Governance Violations (SLA & Fraud) ---
    axs[4].step(time_steps, ping_violations, label="SLA Latency Violation", color="orange", alpha=0.8)
    axs[4].step(time_steps, fraud_detected, label="Fraud/Anomaly Event", color="red", linewidth=2)
    axs[4].set_ylabel("Indicator")
    axs[4].set_title("Instantaneous Governance & Adversarial Events")
    axs[4].legend()
    axs[4].set_ylim(-0.1, 1.1)

    # --- Plot 6: Cumulative Governance Violation Density ---
    cum_ping = np.cumsum(ping_violations)
    cum_jitter = np.cumsum(jitter_violations)
    axs[5].fill_between(time_steps, cum_ping, label="Total SLA Breaches", color="blue", alpha=0.3)
    axs[5].fill_between(time_steps, cum_jitter, label="Total Jitter Breaches", color="cyan", alpha=0.3)
    axs[5].set_ylabel("Cumulative Count")
    axs[5].set_xlabel("Decision Epoch ($t$)")
    axs[5].set_title("System Resilience: Cumulative Constraint Violations")
    axs[5].legend()

    plt.tight_layout()
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
        #full_texts = [
        #    prompt + "\n" + target
        #    for prompt, target in zip(prompts, targets)
        #]

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

            #if step % 10 == 0:
            #    print(f"Epoch {epoch+1}, Step {step}, Loss {loss.item():.4f}")
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


            print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

            avg_val_loss = val_loss / len(val_loader)
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

    # Hugging Face authentication token
    huggingface_token = "hf_something_token"

    # Load the dataset
    data = pd.read_parquet("cellular_dataframe.parquet")
    data["Step"] = range(len(data))  # Add Step column if missing

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
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # 1. Load the skeleton (Base Model)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            quantization_config=quantization_config # load_in_8bit=True
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

        # 4. Handle the embedding resize for your added tokens
        if len(tokenizer) > model.config.vocab_size:
            print(f"Resizing embeddings for {len(tokenizer)} tokens")
            model.resize_token_embeddings(len(tokenizer))


        # Performance tweaks
        model.gradient_checkpointing_enable()

        print("LoRA Adapter and Base Model loaded successfully!")

    except Exception as e:
        print(f"Failed to load: {e}")


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
    aggregate_risk_history, staleness_risks, epistemic_risks, jitter = [], [], [], []
    jitter_history, bt_true, bt_reported, congestion_index = [], [], [], []
    ping_violations_history, jitter_violations_history = [], []
    fraud_events = [] # Binary (1 = Fraud, 0 = No Fraud)
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


    # Stream KPI data from the test dataset and iterate
    for step, kpis in enumerate(stream_kpis_from_dataset(test_data, feed_interval=0.2)): #stream_kpis_real_time(150, feed_interval=0.1)):

        if step >= evaluator["steps"]: break

        fraud_detected = False
        behavior_flagged = False

        # We derive the 'True Confidence' from the jitter.
        # As jitter increases, the 'Ground Truth' certainty decreases.
        base_confidence = 0.95
        jitter_penalty = kpis.get('jitter', 0) / 100.0
        current_bt_true = max(0.1, base_confidence - jitter_penalty)

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
        bt_true.append(kpis.get("bt_true", 1.0))
        bt_reported.append(kpis.get("bt_reported", 1.0))
        congestion_index.append(kpis.get("Traffic Jam Factor", 0))

        # 2. CALCULATE DYNAMIC SLA LIMITS
        sla_p_limit = get_dynamic_threshold(kpis)
        sla_j_limit = get_dynamic_jitter_threshold(kpis)

        # 3. AGENT INFERENCE
        # Passing the pre-formatted 'kpi_input' string to the agent
        for agent in agents:
            try:
                decision = agent.infer(prompt) #kpis["kpi_input"])
                # If decision is a dictionary:
                if isinstance(decision, dict):
                    decision_text = str(decision.get("decision", "")).lower()
                # If decision is just a string:
                else:
                    decision_text = str(decision).lower()

                fraud_detected = fraud_detected or "fraud" in decision_text #decision["decision"].lower()
                behavior_flagged = behavior_flagged or "flagged" in decision_text

            except Exception as e:
                print(f"Inference error: {e}")

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
    extended_visualize_results(
        time_steps=range(len(aggregate_risk_history)),
        aggregate_risks=aggregate_risk_history,
        epistemic_risks=epistemic_risks,
        staleness_risks=staleness_risks,
        congestion_index=congestion_index,
        jitter=jitter_history,
        bt_true=bt_true,
        bt_reported=bt_reported,
        ping_violations=ping_violations_history,
        jitter_violations=jitter_violations_history,
        fraud_detected=fraud_events
    )


if __name__ == "__main__":
    main()
