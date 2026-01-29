import time
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt
import torch
import requests
import pandas as pd
from sklearn.model_selection import train_test_split  # Add this
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, default_data_collator
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, DatasetDict  # Simplify creating and loading datasets
from torchvision import models

from accelerate import Accelerator
from torch.utils.data import DataLoader


# --- KPI Stream Simulation ---
# TODO:: revisit me to pick up my KPIs from some system

def stream_kpis_real_time(steps=150, feed_interval=1):
    """
    Simulate or fetch KPI data and stream it to the LLM in real time.
    :param steps: Number of data points to generate
    :param feed_interval: Time interval (in seconds) between each step
    """
    t = np.arange(steps)
    drift_levels = 0.05 + 0.1 * np.random.rand(steps)
    bt_true = 0.95 * np.ones(steps)
    
    drift_levels[50:100] += 0.5  # Simulated drift in jitter
    bt_true[50:100] -= 0.3      # Degraded confidence

    # Inject fake reporting as an anomaly
    bt_reported = bt_true.copy()
    bt_reported[100:150] = 0.9  # Lie about confidence at later steps

    for i in range(steps):
        yield {
            "step": i,
            "bt_true": bt_true[i],
            "bt_reported": bt_reported[i],
            "latency": drift_levels[i]
        }
        time.sleep(feed_interval)  # Simulate real-time streaming


#--- Fine-Tuning Functions ---
def generate_finetuning_dataset():
    """
    Generate a fine-tuning dataset for communication systems KPI tasks.
    """
    # Example synthetic dataset for GPT fine-tuning.
    data = [
        {
            "kpi_input": f"Latency: {latency}ms, Jitter: {jitter}ms, Packet Loss: {packet_loss}%",
            "kpi_description": f"The system is {'critical' if latency > 500 else 'stable'}, "
                                f"jitter is {jitter}ms, Packet Loss is {packet_loss}%."
        }
        for latency, jitter, packet_loss in [
            (120, 20, 1),
            (800, 50, 0),
            (500, 45, 5),
            (100, 10, 0),
            (900, 200, 10)
        ]
    ]
    return Dataset.from_list(data)


def stream_kpis_from_dataset(df, feed_interval=1):
    """
    Stream KPI data dynamically row by row from a dataframe.
    :param df: Pandas DataFrame containing the KPI data.
    :param feed_interval: Time interval (in seconds) between each step.
    """
    for timestamp, row in df.iterrows():
        yield {
            "device": row["device"],
            "ping_ms": row["ping_ms"],
            "datarate": row["datarate"],
            "jitter": row["jitter"],
            "target_datarate": row["target_datarate"],
            "direction": row["direction"],
            "measured_qos": row["measured_qos"],
            "operator": row["operator"],
            "step": timestamp,  # Use the timestamp as the identifier or step
        }
        time.sleep(feed_interval)

# TODO:: use NetGPT here
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


    def old_infer(self, kpis):
        """
        Use the LLM to infer decisions based on KPIs and past context.
        Args:
            kpis: Dictionary containing network KPI data.

        Returns:
            dict: Recommended decisions and insights based on KPIs.
        """
        # Build a prompt with context from history for the most recent 10 entries
        # context = "\n".join([f"Step {entry['step']}: {entry['decision']}" for entry in self.history[-10:]])
        #prompt = f"""
        #Using the KPI context and the following current KPI values:
        #Reported Confidence: {kpis['bt_reported']:.2f}
        #True Confidence: {kpis['bt_true']:.2f}
        #Latency: {kpis['latency']:.2f}

        #Previous decisions:
        #{context}

        #Instructions:
        #1. Classify the risk level as "low", "moderate", "high", or "critical".
        #2. Predict if fraud exists (True/False) and describe why.
        #3. Suggest governance credits adjustments or fraud penalties.
        #4. Recommend premium adjustments or other operational decisions.
        #"""

        # Build a prompt
        prompt = f"""
        Current KPI values:
        - Reported Confidence: {kpis['bt_reported']:.2f}
        - True Confidence: {kpis['bt_true']:.2f}
        - Latency: {kpis['latency']:.2f}

        Suggest:
        1. Risk classification ("low", "moderate", "high", or "critical").
        2. Fraud detection: Is fraud occurring? (True/False).
        3. Adjustments for governance credits or penalties.
        """

        # Generate response from the LLM
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            # inputs = {key: value.to(self.model.device) for key, value in inputs.items()}  # Place on model's device
            # outputs = self.model.generate(**inputs, max_new_tokens=150)

            inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move input to GPU or CPU

            with torch.no_grad():  # Disable gradient tracking to save memory
                outputs = self.model.generate(**inputs, max_new_tokens=150)
                decision_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Release unused GPU memory using `torch.cuda.empty_cache`
            if self.device == "cuda":
                torch.cuda.empty_cache()

            # Store the decision in history
            decision = {"step": kpis["step"], "decision": decision_text}
            # Process raw model output into a structured response
            return self._parse_decision(decision_text)

            return decision
        except Exception as e:
            # Explicitly clear memory upon failure
            if self.device == "cuda":
                torch.cuda.empty_cache()
            print(f"{self.name}: Failed to generate inference: {e}")
            raise RuntimeError(f"Inference failure in agent {self.name}: {e}")

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
        """

        try:
            # Tokenize the input prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

            # Move tokenized inputs to the same device as the model
            inputs = {key: value.to(self.device) for key, value in inputs.items()}  # Align inputs to model's device

            # Generate the response using the model
            with torch.no_grad():  # Disable gradient calculations for inference
                outputs = self.model.generate(**inputs, max_new_tokens=150)

            # Decode the response
            decision_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Parse the decision into a structured format
            return self._parse_decision(decision_text)

        except Exception as e:
            print(f"{self.name}: Failed to generate inference: {e}")
            raise RuntimeError(f"Inference failure in agent {self.name}: {e}")


    def _parse_decision(self, decision_text):
        """
        Extract structured data from the raw model output.
        Args:
            decision_text: The raw generation from the LLM.
        Returns:
            dict: Parsed data including risk, fraud detection, and adjustments.
        """
        try:
            # Using regex or string parsing for simple interpretation
            risk_match = re.search(r"Risk classification: (\w+)", decision_text, re.IGNORECASE)
            fraud_match = re.search(r"Fraud occurring: (\w+)", decision_text, re.IGNORECASE)
            governance_match = re.search(r"Adjustments: (.+)", decision_text, re.IGNORECASE)

            # Parse structured values
            risk = risk_match.group(1) if risk_match else "Unknown"
            fraud = fraud_match.group(1).lower() == "true" if fraud_match else False
            governance = governance_match.group(1).strip() if governance_match else "No adjustments specified."

            return {
                "risk_classification": risk.capitalize(),
                "fraud_detected": fraud,
                "governance_adjustments": governance
            }
        except Exception as e:
            print(f"{self.name}: Failed to parse decision: {e}")
            return {
                "risk_classification": "Error",
                "fraud_detected": "Error",
                "governance_adjustments": str(e)
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


# --- Insurance Premium Logic ---

def calculate_risk_factors(metadata, evaluator, step, fraud_detected, behavior_flagged):
    """
    Calculate epistemic, environmental, staleness, and additional penalties or rewards.
    Args:
        metadata: Dictionary containing row data (KPI values).
        evaluator: Configuration for risk calculations.
        step: Current step in the streaming pipeline.
        fraud_detected (bool): Whether fraud was detected.
        behavior_flagged (bool): Whether anomalous behavior was flagged.

    Returns:
        dict: Dictionary containing the calculated risk factors.
    """
    # Extract relevant KPI values
    latency = metadata['ping_ms']
    jitter = metadata['jitter']
    datarate = metadata['datarate']
    target_datarate = metadata['target_datarate']
    reported_quality = metadata['PCell_RSRQ_1']

    # Calculate the risks
    if target_datarate > 0:
        epistemic_risk = evaluator["gamma"] * ((target_datarate - datarate) / target_datarate)
    else:
        epistemic_risk = 0

    network_risk = evaluator["beta"] * jitter
    staleness_risk = evaluator["delta"] * max(0, evaluator["dt_req"] - latency)
    congestion_index = metadata["Traffic Jam Factor"]

    fraud_penalty = 20.0 if fraud_detected else 0.0
    behavior_penalty = 15.0 if behavior_flagged else 0.0

    return {
        "epistemic": max(epistemic_risk, 0),
        "network": network_risk,
        "staleness": staleness_risk,
        "congestion_index": congestion_index,  # Include congestion index for visualization
        "fraud_penalty": fraud_penalty,
        "behavior_penalty": behavior_penalty,
    }


def premium_engine_extended(risk_factors, evaluator, contract, governance_credit=0.25):
    """
    Calculate the extended insurance premium.
    """
    total_risk_premium = (
        evaluator["p_base"]
        + risk_factors["epistemic"]
        + risk_factors["network"]
        + risk_factors["staleness"]
        + risk_factors["fraud_penalty"]
        + risk_factors["behavior_penalty"]
    )

    # Apply governance credits if fraud is absent
    if governance_credit > 0:
        total_risk_premium *= (1 - governance_credit)

    # Apply uncovered penalties if the premium exceeds the cap
    if total_risk_premium > contract["premium_cap"]:
        total_risk_premium *= contract["uncovered_penalty"]

    return total_risk_premium


def extended_visualize_results(
    time_steps,  # List of time steps
    epistemic_risks,  # Epistemic risk values
    staleness_risks,  # Staleness risk values
    congestion_index,  # Congestion index values
    signal_quality,  # Dict with signal quality metrics (RSRP, RSRQ, SNR)
    fraud_detected,  # Binary array indicating fraud detection events (1 = fraud, 0 = no fraud)
    sla_violations=None,  # Optional dict with SLA violation counts for each KPI
    jitter=None,  # Optional jitter values (for comparison)
    kpi_variability=None  # Optional variability of KPIs
):
    """
    Static visualization of risk metrics, signal quality, and detection indicators.
    Args:
        time_steps: List of time steps for the plots.
        epistemic_risks: List of epistemic risk values.
        staleness_risks: List of staleness risk values.
        congestion_index: List of congestion index values.
        signal_quality: Dict containing RSRP, RSRQ, and SNR signal quality time-series.
        fraud_detected: List of binary values indicating fraud detection (1 = fraud, 0 = no fraud).
        sla_violations: Dictionary containing SLA violation counts for each KPI.
        jitter: Optional jitter values over time.
        kpi_variability: Optional dictionary containing variability (e.g., std) for KPIs.
    """
    import matplotlib.pyplot as plt

    # Set up a figure with subplots
    fig, axs = plt.subplots(6, 1, figsize=(16, 24), sharex=True)

    # Plot 1: Epistemic Risks
    axs[0].plot(time_steps, epistemic_risks, label="Epistemic Risk", color="purple", linewidth=2)
    axs[0].set_ylabel("Epistemic Risk")
    axs[0].set_title("Epistemic Risk Over Time")
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: Staleness Risks
    axs[1].plot(time_steps, staleness_risks, label="Staleness Risk", color="orange", linewidth=2)
    axs[1].set_ylabel("Staleness Risk")
    axs[1].set_title("Staleness Risk Over Time")
    axs[1].legend()
    axs[1].grid(True)

    # Plot 3: Congestion Index
    axs[2].plot(time_steps, congestion_index, label="Congestion Index", color="brown", linewidth=2)
    axs[2].set_ylabel("Congestion Index")
    axs[2].set_title("Traffic Congestion Index Over Time")
    axs[2].legend()
    axs[2].grid(True)

    # Plot 4: Signal Quality Metrics (RSRP, RSRQ, SNR)
    axs[3].plot(time_steps, signal_quality["RSRP"], label="RSRP", color="blue", alpha=0.8)
    axs[3].plot(time_steps, signal_quality["RSRQ"], label="RSRQ", color="green", alpha=0.8)
    axs[3].plot(time_steps, signal_quality["SNR"], label="SNR", color="orange", alpha=0.8)
    axs[3].set_title("Signal Quality Metrics Over Time")
    axs[3].set_ylabel("Signal Metrics")
    axs[3].legend()
    axs[3].grid(True)

    # Plot 5: Fraud Detection Events (Binary)
    axs[4].step(time_steps, fraud_detected, label="Fraud Detected", color="red", linewidth=2, where="post")
    axs[4].set_ylabel("Fraud Indicator")
    axs[4].set_title("Fraud Detection Over Time")
    axs[4].legend()
    axs[4].grid(True)

    # Plot 6: SLA Violations (if provided)
    if sla_violations:
        keys = list(sla_violations.keys())
        values = list(sla_violations.values())
        axs[5].bar(keys, values, color="blue", alpha=0.7)
        axs[5].set_ylabel("Violation Count")
        axs[5].set_title("SLA Violations by KPI")
        axs[5].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # Optional Variability Plot for KPIs
    if kpi_variability:
        plt.figure(figsize=(12, 6))
        keys = list(kpi_variability.keys())
        values = list(kpi_variability.values())
        plt.bar(keys, values, color="blue", alpha=0.7)
        plt.title("KPI Variability (Standard Deviation)")
        plt.xlabel("KPI")
        plt.ylabel("Variability")
        plt.grid(True)
        plt.show()

def old_extended_visualize_results(premium_history, jitter, bt_true, bt_reported, staleness_risks, epistemic_risks, congestion_index):
    """
    Static visualization function to display all key performance metrics after processing.
    Args:
        premium_history: List of calculated premiums over time.
        jitter: List of jitter values over time.
        bt_true: List of true confidence values over time.
        bt_reported: List of reported confidence values over time.
        staleness_risks: List of staleness risk contributions over time.
        epistemic_risks: List of epistemic risk contributions over time.
        congestion_index: List of congestion index values over time.
    """
    # Ensure all inputs are of the same length
    min_len = min(len(premium_history), len(jitter), len(bt_true), len(bt_reported),
                  len(staleness_risks), len(epistemic_risks), len(congestion_index))
    premium_history = premium_history[:min_len]
    jitter = jitter[:min_len]
    bt_true = bt_true[:min_len]
    bt_reported = bt_reported[:min_len]
    staleness_risks = staleness_risks[:min_len]
    epistemic_risks = epistemic_risks[:min_len]
    congestion_index = congestion_index[:min_len]

    # Create the figure and axis layout
    fig, axs = plt.subplots(5, 1, figsize=(16, 24), sharex=True)

    # Plot 1: Premium Dynamics
    axs[0].plot(premium_history, color="crimson", linewidth=2, label="Premium ($)")
    axs[0].set_ylabel("Premium ($)")
    axs[0].set_title("6G AI Insurance Premium History")
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: Metadata and KPI Trends
    axs[1].plot(bt_true, label="True Confidence", linestyle="--", alpha=0.7, color="blue")
    axs[1].plot(bt_reported, label="Reported Confidence", linewidth=2, color="green")
    axs[1].fill_between(range(len(jitter)), 0, 1.5, where=(np.array(jitter) > 0.4),
                        color='gray', alpha=0.3, label="High Jitter Zone")
    axs[1].set_ylabel("Confidence / Jitter")
    axs[1].legend()
    axs[1].grid(True)

    # Plot 3: Staleness Risk Contribution
    axs[2].plot(staleness_risks, label="Staleness Risk", color="orange")
    axs[2].set_ylabel("Staleness Risk")
    axs[2].legend()
    axs[2].grid(True)

    # Plot 4: Epistemic Risk Contribution
    axs[3].plot(epistemic_risks, label="Epistemic Risk", color="purple")
    axs[3].set_ylabel("Epistemic Risk")
    axs[3].legend()
    axs[3].grid(True)

    # Plot 5: Congestion Index
    axs[4].plot(congestion_index, label="Congestion Index", color="brown")
    axs[4].set_ylabel("Congestion Index")
    axs[4].set_xlabel("Time Steps")
    axs[4].legend()
    axs[4].grid(True)

    # Adjust layout for better visuals
    plt.tight_layout()
    plt.show()


from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from torch.optim import AdamW 
from torch.utils.data import DataLoader
from accelerate import Accelerator
# from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

# --- Fine-Tuning ---
def fine_tune_model(train_data, base_model_name, tokenizer, val_data=None, save_dir="./finetuned_model"):
    """
    Fine-tune a base model with a training dataset using LoRA (Low-Rank Adaptation).
    Args:
        train_data: Pandas DataFrame containing the training data.
        base_model_name: Pretrained model name (e.g., 'EleutherAI/gpt-neo-1.3B' or 'NetGPT').
        tokenizer: Tokenizer for the model.
        val_data: Pandas DataFrame containing the validation data (optional).
        save_dir: Directory to save the fine-tuned model.
    """
    
    accelerator = Accelerator()
    print("Preprocessing the training dataset for fine-tuning...")

    # Convert to HF datasets
    hf_train_dataset = Dataset.from_pandas(train_data)
    hf_val_dataset = Dataset.from_pandas(val_data) if val_data is not None else None

    # Explicitly add a pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # ✅ CORRECT batched preprocessing
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

        #tokenized['labels'] = tokenized['input_ids'].copy()  # Copy input IDs into labels

        # Apply masking for the prompt tokens
        #for i, prompt in enumerate(prompts):
        #    prompt_tokens = tokenizer(
        #        prompt,
        #        truncation=True,
        #        max_length=512,
        #        padding=True,
        #    )["input_ids"]

        #    num_prompt_tokens = len(prompt_tokens)
        #    label = tokenized['labels'][i]

        #    # Mask the prompt part (set to -100 for ignored labels during loss calculation)
        #    label[:num_prompt_tokens] = [-100] * num_prompt_tokens
        #    tokenized['labels'][i] = label

        #tokenized["labels"] = labels
        #return tokenized


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

    # ✅ Correct collator for supervised causal LM
    #data_collator = DataCollatorWithPadding(
    #    tokenizer=tokenizer,
    #    pad_to_multiple_of=8,
    #)

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
    # CRITICAL: resize embeddings
    base_model.resize_token_embeddings(len(tokenizer))

    # Ensure model knows pad token
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # DEBUG code
    print("Tokenizer vocab size:", len(tokenizer))
    print("Model vocab size:", base_model.config.vocab_size)
    print("Pad token id:", tokenizer.pad_token_id)

    batch = next(iter(train_loader))
    print("Max input_id:", batch["input_ids"].max().item())  

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

    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)  

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

            if step % 10 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss {loss.item():.4f}")

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(**batch)
                    val_loss += outputs.loss.item()
            print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

    #print("Saving fine-tuned model...")
    #accelerator.wait_for_everyone()
    #model.save_pretrained("./lora_finetuned_model", save_function=accelerator.save)
    #tokenizer.save_pretrained("./lora_finetuned_tokenizer")

    #return "./lora_finetuned_model"
    # Save Fine-Tuned Model
    save_path = f"{save_dir}/{base_model_name.replace('/', '_')}"
    print(f"Saving fine-tuned model to {save_path}...")
    accelerator.wait_for_everyone()
    model.save_pretrained(save_path, save_function=accelerator.save)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}.")

    return save_path


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
    huggingface_token = "my_hf_token"

    # Load the dataset
    data = pd.read_parquet("cellular_dataframe.parquet")
    data["Step"] = range(len(data))  # Add Step column if missing
    
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

    ##fine_tune = True  # Set to False if fine-tuning is not required

    # Check for GPU availability, else fallback to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the model/tokenizer once and share across agents
    # model_name = "meta-llama/Llama-2-7b-hf" 
    ##base_model_name = "EleutherAI/gpt-neo-1.3B"  # Base model
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #model = AutoModelForCausalLM.from_pretrained(model_name)

    ##tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    ##model_name_or_path = base_model_name

    # Preprocess and fine-tune the model
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M") # EleutherAI/gpt-neo-1.3B") # 
    model_name = fine_tune_model(train_data, "EleutherAI/gpt-neo-125M", tokenizer, val_data) # EleutherAI/gpt-neo-125M


    try:
        print("Loading shared tokenizer and model...")
        ##model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        ##tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Load fine-tuned model
        print(f"Loading fine-tuned model from {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # Automatically map layers to device (e.g., CPU/GPU)
            load_in_8bit=True  # Load model in 8-bit precision for memory efficiency
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print("Shared Model and Tokenizer Loaded successfully!")
    except Exception as e:
        print(f"Failed to load model/tokenizer: {e}")
        return

    # Performance tweaks
    model.gradient_checkpointing_enable()

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
    premium_history, staleness_risks, epistemic_risks, jitter = [], [], [], []
    bt_true, bt_reported, congestion_index = [], [], []

    # Store risk factors for fraud and behavioral analysis
    fraud_flags = []
    behavior_flags = []
    signal_quality = {"RSRP": [], "RSRQ": [], "SNR": []}
    fraud_events = []  # Binary (1 = Fraud, 0 = No Fraud)
    sla_violations = {"ping_ms": 0, "jitter": 0, "datarate": 0, "target_datarate": 0}
    #update_plot = dynamic_visualization()

    # Stream KPI data (can integrate with real-time production APIs)
    # TODO:: pick KPIs from real api source - Done below

    # Stream KPI data from the test dataset and iterate
    for step, kpis in enumerate(stream_kpis_from_dataset(test_data, feed_interval=0.2)): #stream_kpis_real_time(150, feed_interval=0.1)):
        fraud_detected = False
        behavior_flagged = False

        jitter.append(kpis["latency"])
        bt_true.append(kpis["bt_true"])
        bt_reported.append(kpis["bt_reported"])
        congestion_index.append(kpis["congestion"])

        # Let all LLM agents infer decisions for the current KPI data
        for agent in agents:
            try:
                decision = agent.infer(kpis)
                #print(f"Agent Decision for Step {kpis['step']} ({agent.name}): {decision['decision']}")
                # Print the structured decision report
                print_decision(step, agent.name, decision)

                # Check if fraud or flagged behavior was detected from any agent
                fraud_detected = fraud_detected or "fraud" in decision["decision"].lower()
                behavior_flagged = behavior_flagged or "flagged" in decision["decision"].lower()
            except Exception as e:
                print(f"Error during agent decision on Step {kpis['step']}: {e}")

        torch.cuda.empty_cache()  # Clear memory after each step

        # Calculate risk factors based on input KPIs, fraud, and behavioral flags
        risk_factors = calculate_risk_factors(
            metadata=kpis,
            evaluator=evaluator,
            step=kpis["step"],
            fraud_detected=fraud_detected,
            behavior_flagged=behavior_flagged
        )

        # Append individual risks
        staleness_risks.append(risk_factors["staleness"])
        epistemic_risks.append(risk_factors["epistemic"])
        signal_quality["RSRP"].append(kpis["PCell_RSRP_1"])
        signal_quality["RSRQ"].append(kpis["PCell_RSRQ_1"])
        signal_quality["SNR"].append(kpis["PCell_SNR_1"])

        # Update violations
        sla_violations["ping_ms"] += 1 if kpis["ping_ms"] > SLA_PING_THRESHOLD else 0
        sla_violations["jitter"] += 1 if kpis["jitter"] > SLA_JITTER_THRESHOLD else 0

        # Calculate the premium based on the extended premium logic
        #premium = premium_engine_extended(
        #    risk_factors,
        #    evaluator,
        #    contract,
        #    governance_credit=0.25 if not fraud_detected else 0  # Reward governance credits if no fraud detected
        #)
        #premium_history.append(premium)

        # Track fraud and behavior flags
        fraud_flags.append(fraud_detected)
        behavior_flags.append(behavior_flagged)

        #print(f"Step {kpis['step']}: Premium = ${premium:.2f}, Risk Factors = {risk_factors}")

    
    # Call the visualization function after streaming is done
    #extended_visualize_results(premium_history, jitter, bt_true, bt_reported, staleness_risks, epistemic_risks, congestion_index)

    extended_visualize_results(
        time_steps=range(len(epistemic_risks)),
        epistemic_risks=epistemic_risks,
        staleness_risks=staleness_risks,
        congestion_index=congestion_index,
        signal_quality=signal_quality,
        fraud_detected=fraud_events,
        sla_violations=sla_violations,
    )

if __name__ == "__main__":
    main()
