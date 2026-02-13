"""
Metric calculation utilities for GIRAF.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math


def calculate_confidence_gap(model, tokenizer, prompt, ground_truth_val):
    """
    Calculate epistemic uncertainty and confidence metrics.
    
    Args:
        model: The LLM model
        tokenizer: Model tokenizer
        prompt: Input prompt
        ground_truth_val: Ground truth value
        
    Returns:
        tuple: (epistemic_risk, reported_confidence, prediction)
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
        
        reported_conf = torch.max(probs).item()
        
        prediction = model.generate(
            **inputs,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )
        
        decoded_prediction = tokenizer.decode(prediction[0], skip_special_tokens=True)
        prediction_matches_truth = ground_truth_val in decoded_prediction if isinstance(ground_truth_val, str) else True
        accuracy = 1.0 if prediction_matches_truth else 0.0
        
        r_epi = abs(reported_conf - accuracy)
        
        return r_epi, reported_conf, decoded_prediction
        
    except Exception as e:
        print(f"Inference error: {e}")
        return 1.0, 0.5, "error"


def get_dynamic_threshold(kpis, base_req=40.0, safety_floor=10.0):
    """
    Compute dynamic latency SLA based on speed and congestion.
    
    Args:
        kpis (dict): KPI data
        base_req (float): Base requirement
        safety_floor (float): Minimum threshold
        
    Returns:
        float: Dynamic threshold
    """
    v = kpis.get("speed_kmh", 0)
    c = kpis.get("Traffic Jam Factor", 0)
    v_ref = 120.0
    
    dynamic_limit = base_req * (1 - (c/10)) * math.exp(-v / v_ref) + safety_floor
    return max(dynamic_limit, safety_floor)


def get_dynamic_jitter_threshold(kpis, nominal_jitter=15.0, floor=2.0):
    """
    Compute dynamic jitter SLA based on SNR and congestion.
    
    Args:
        kpis (dict): KPI data
        nominal_jitter (float): Nominal jitter value
        floor (float): Minimum threshold
        
    Returns:
        float: Dynamic jitter threshold
    """
    snr = kpis.get("PCell_SNR_1", 20)
    c = kpis.get("Traffic Jam Factor", 0)
    snr_max = 30.0
    
    dynamic_jitter_limit = nominal_jitter * (snr / snr_max) * math.exp(-0.5 * (c/10)) + floor
    return max(dynamic_jitter_limit, floor)


def get_baseline_and_giraf_confidence(model, tokenizer, prompt, bt_true):
    """
    Extract both pretrained and GIRAF-governed confidence.
    
    Args:
        model: LLM model
        tokenizer: Model tokenizer
        prompt: Input prompt
        bt_true: Ground truth confidence
        
    Returns:
        tuple: (pretrained_confidence, giraf_confidence)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    logits = outputs.scores[0]
    probs = F.softmax(logits, dim=-1)
    pretrained_conf = torch.max(probs, dim=-1).values.item()
    
    r_epi = abs(bt_true - pretrained_conf)
    giraf_conf = pretrained_conf * (1 - r_epi)
    
    return pretrained_conf, giraf_conf
