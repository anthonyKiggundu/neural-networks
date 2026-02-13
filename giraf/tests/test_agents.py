"""
Unit tests for LLM agents.
"""

import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from giraf.agents import LLMKPIAgent


@pytest.fixture
def mock_model_and_tokenizer():
    """Fixture for model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def test_agent_initialization(mock_model_and_tokenizer):
    """Test agent initialization."""
    model, tokenizer = mock_model_and_tokenizer
    
    agent = LLMKPIAgent(
        name="Test Agent",
        drift_tag="test",
        tokenizer=tokenizer,
        model=model,
        device="cpu"
    )
    
    assert agent.name == "Test Agent"
    assert agent.drift_tag == "test"
    assert agent.device == "cpu"


def test_agent_requires_model_and_tokenizer():
    """Test that agent requires both model and tokenizer."""
    with pytest.raises(ValueError):
        LLMKPIAgent(name="Test", tokenizer=None, model=None)


def test_agent_inference(mock_model_and_tokenizer):
    """Test agent inference."""
    model, tokenizer = mock_model_and_tokenizer
    
    agent = LLMKPIAgent(
        name="Test Agent",
        tokenizer=tokenizer,
        model=model,
        device="cpu"
    )
    
    mock_kpis = {
        'device': 'test_device',
        'name': 0,
        'Latitude': 0.0,
        'Longitude': 0.0,
        'Altitude': 0.0,
        'speed_kmh': 50,
        'Traffic Jam Factor': 5,
        'ping_ms': 20,
        'jitter': 5,
        'datarate': 100,
        'target_datarate': 100,
        'PCell_RSRP_1': -80,
        'PCell_RSRQ_1': -10,
        'PCell_SNR_1': 20,
        'PCell_Downlink_Num_RBs': 50,
        'PCell_Uplink_Num_RBs': 25,
        'measured_qos': 0.9,
        'operator': 'test_op'
    }
    
    decision = agent.infer(mock_kpis)
    
    assert isinstance(decision, dict)
    assert 'decision' in decision
    assert 'fraud_detected' in decision
    assert 'risk_classification' in decision
