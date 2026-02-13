"""
Unit tests for risk calculation.
"""

import pytest
from giraf.evaluation import calculate_risk_factors
from giraf.config import GIRAFConfig


def test_risk_calculation_basic():
    """Test basic risk calculation."""
    config = GIRAFConfig()
    
    metadata = {
        "bt_true": 0.9,
        "Traffic Jam Factor": 5,
        "ping_ms": 30,
        "constraint_coverage": 0.8
    }
    
    risk_data = calculate_risk_factors(
        metadata=metadata,
        evaluator=config.DEFAULT_EVALUATOR,
        step=0,
        fraud_detected=False,
        behavior_flagged=False,
        threshold=45.0
    )
    
    assert "aggregate_risk" in risk_data
    assert "epistemic_component" in risk_data
    assert "staleness_component" in risk_data
    assert "environmental_component" in risk_data
    assert "mitigation_signal" in risk_data
    assert "trust_score" in risk_data


def test_risk_with_fraud_penalty():
    """Test that fraud detection increases risk."""
    config = GIRAFConfig()
    
    metadata = {
        "bt_true": 0.9,
        "Traffic Jam Factor": 5,
        "ping_ms": 30,
        "constraint_coverage": 0.8
    }
    
    risk_no_fraud = calculate_risk_factors(
        metadata, config.DEFAULT_EVALUATOR, 0, False, False
    )
    
    risk_with_fraud = calculate_risk_factors(
        metadata, config.DEFAULT_EVALUATOR, 0, True, False
    )
    
    assert risk_with_fraud["aggregate_risk"] > risk_no_fraud["aggregate_risk"]


def test_mitigation_signal_threshold():
    """Test mitigation signal triggers above threshold."""
    config = GIRAFConfig()
    
    # High risk scenario
    high_risk_metadata = {
        "bt_true": 0.3,  # Low confidence
        "Traffic Jam Factor": 10,  # High congestion
        "ping_ms": 200,  # High latency
        "constraint_coverage": 0.5
    }
    
    risk_data = calculate_risk_factors(
        high_risk_metadata, config.DEFAULT_EVALUATOR, 0, True, True, threshold=45.0
    )
    
    assert risk_data["mitigation_signal"] == 1
