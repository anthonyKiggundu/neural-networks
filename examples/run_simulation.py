"""
Main simulation script for GIRAF framework.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from giraf import GIRAFConfig, LLMKPIAgent
from giraf.data import stream_kpis_from_dataset, prepare_dataset
from giraf.evaluation import (
    calculate_risk_factors,
    calculate_confidence_gap,
    get_dynamic_threshold,
    get_dynamic_jitter_threshold,
    get_baseline_and_giraf_confidence
)
from giraf.training import fine_tune_model
from giraf.visualization import (
    extended_visualize_results,
    plot_risk_distribution_by_traffic_jam_factor,
    plot_verification_staleness_dist,
    generate_comparative_reliability_diagram
)


def main():
    """Main simulation entry point."""
    
    # Initialize configuration
    config = GIRAFConfig()
    evaluator = config.DEFAULT_EVALUATOR
    
    # Load dataset
    print("Loading dataset...")
    data = pd.read_parquet("cellular_dataframe.parquet")
    data["Step"] = range(len(data))
    
    # Prepare data splits
    train_data, val_data, test_data = prepare_dataset(
        data,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO
    )
    
    # Calculate dynamic thresholds
    SLA_PING_THRESHOLD = np.nanpercentile(test_data['ping_ms'], config.SLA_PING_PERCENTILE)
    SLA_JITTER_THRESHOLD = np.nanpercentile(test_data['jitter'], config.SLA_JITTER_PERCENTILE)
    
    if np.isnan(SLA_PING_THRESHOLD):
        SLA_PING_THRESHOLD = config.DEFAULT_SLA_PING
    if np.isnan(SLA_JITTER_THRESHOLD):
        SLA_JITTER_THRESHOLD = config.DEFAULT_SLA_JITTER
    
    print(f"Dynamic Thresholds: Ping={SLA_PING_THRESHOLD:.2f}ms, Jitter={SLA_JITTER_THRESHOLD:.2f}ms")
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Fine-tune model
    print("Fine-tuning model...")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
    model_path = fine_tune_model(
        train_data,
        config.BASE_MODEL_NAME,
        tokenizer,
        val_data,
        max_steps=config.MAX_TRAINING_STEPS
    )
    
    # Load model
    print("Loading fine-tuned model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if len(tokenizer) > base_model.config.vocab_size:
        print(f"Resizing embeddings: {base_model.config.vocab_size} -> {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))
    
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    
    model.gradient_checkpointing_enable()
    
    # Initialize agent
    agents = [
        LLMKPIAgent(
            drift_tag=str(i),
            name=f"KPI Agent {i}",
            tokenizer=tokenizer,
            model=model,
            device=device
        ) for i in range(1)
    ]
    
    # Initialize tracking variables
    aggregate_risk_history = []
    staleness_risks = []
    epistemic_risks = []
    jitter_history = []
    bt_true_history = []
    bt_reported_history = []
    congestion_index = []
    accuracy_history = []
    ping_violations_history = []
    fraud_events = []
    lv_history = []
    smt_depth_history = []
    pretrained_conf_history = []
    giraf_conf_history = []
    
    # Run simulation
    print("\nStarting simulation...")
    for step, kpis in enumerate(stream_kpis_from_dataset(test_data, feed_interval=0.2)):
        if step >= evaluator["steps"]:
            break
        
        fraud_detected = False
        behavior_flagged = False
        
        # Calculate ground truth confidence
        base_confidence = 0.95
        snr_val = kpis.get('PCell_SNR_1', 20)
        jitter_val = kpis.get('jitter', 0.5)
        current_bt_true = np.exp(-config.LAMBDA_SENSITIVITY * (jitter_val / max(1, snr_val)))
        
        bt_true_history.append(current_bt_true)
        kpis['bt_true'] = current_bt_true
        
        prompt = kpis.get('kpi_input', "")
        
        current_ping = kpis.get("ping_ms", 0)
        current_jitter = kpis.get("jitter", 0)
        jitter_history.append(current_jitter)
        congestion_index.append(kpis.get("Traffic Jam Factor", 0))
        
        # Dynamic SLA limits
        sla_p_limit = get_dynamic_threshold(kpis)
        sla_j_limit = get_dynamic_jitter_threshold(kpis)
        
        step_confidences = []
        step_preds = []
        
        # Agent inference
        for agent in agents:
            try:
                r_epi, reported_conf, actual_conf = calculate_confidence_gap(
                    model, tokenizer, prompt, kpis['kpi_description']
                )
                step_confidences.append(reported_conf)
                
                kpis['bt_true'] = current_bt_true
                kpis['bt_reported'] = reported_conf
                kpis['r_epi'] = r_epi
                
                decision = agent.infer(prompt)
                
                if isinstance(decision, dict):
                    decision_text = str(decision.get("decision", "")).lower()
                else:
                    decision_text = str(decision).lower()
                
                fraud_detected = fraud_detected or "fraud" in decision_text
                behavior_flagged = behavior_flagged or "flagged" in decision_text
                
                risk_data = calculate_risk_factors(
                    kpis, evaluator, step, fraud_detected, behavior_flagged,
                    threshold=config.MITIGATION_THRESHOLD
                )
                
                step_preds.append(reported_conf)
                
                aggregate_risk_history.append(risk_data["aggregate_risk"])
                epistemic_risks.append(risk_data["epistemic_component"])
                staleness_risks.append(risk_data["staleness_component"])
                lv_history.append(risk_data.get("lv", 0))
                smt_depth_history.append(risk_data.get("smt_depth", 8))
                
                ping_violations_history.append(1 if kpis.get("ping_ms", 0) > SLA_PING_THRESHOLD else 0)
                fraud_events.append(1 if fraud_detected else 0)
                
            except Exception as e:
                print(f"Inference error: {e}")
                step_confidences.append(0.5)
        
        # Record confidence data
        avg_reported_conf = np.mean(step_confidences) if step_confidences else 0.5
        raw_llm_conf, giraf_conf = get_baseline_and_giraf_confidence(
            model, tokenizer, prompt, current_bt_true
        )
        
        actual_success = 1 if np.abs(current_bt_true - giraf_conf) < config.TRUST_THRESHOLD else 0
        
        bt_reported_history.append(avg_reported_conf)
        accuracy_history.append(actual_success)
        giraf_conf_history.append(giraf_conf)
        pretrained_conf_history.append(raw_llm_conf)
        
        torch.cuda.empty_cache()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    tau = 0.2
    time_span = [step * tau for step in range(len(aggregate_risk_history))]
    
    extended_visualize_results(
        time_span=time_span,
        aggregate_risks=aggregate_risk_history,
        epistemic_risks=epistemic_risks,
        staleness_risks=staleness_risks,
        congestion_index=congestion_index,
        jitter=jitter_history,
        bt_true=bt_true_history,
        bt_reported=bt_reported_history,
        ping_violations=ping_violations_history,
        jitter_violations=[0] * len(ping_violations_history),
        fraud_detected=fraud_events
    )
    
    if bt_reported_history and accuracy_history:
        generate_comparative_reliability_diagram(
            y_true=accuracy_history,
            giraf_preds=giraf_conf_history,
            pretrained_preds=pretrained_conf_history
        )
    
    plot_risk_distribution_by_traffic_jam_factor(
        congestion_data=congestion_index,
        epistemic_risks=epistemic_risks,
        staleness_risks=staleness_risks
    )
    
    plot_verification_staleness_dist(
        lv_history, smt_depth_history, sla_deadline=evaluator["dt_req"]
    )
    
    # Print statistics
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    
    risk_arr = np.array(aggregate_risk_history)
    mean_risk = risk_arr[~np.isnan(risk_arr)]
    peak_risk = np.max(risk_arr)
    
    
