"""
Unit tests for configuration.
"""

import pytest
from giraf.config import GIRAFConfig


def test_config_defaults():
    """Test default configuration values."""
    config = GIRAFConfig()
    
    assert config.EPOCH_DURATION_MS == 100
    assert config.MITIGATION_THRESHOLD == 45.0
    assert config.TRUST_THRESHOLD == 0.15
    assert config.BASE_MODEL_NAME == "EleutherAI/gpt-neo-125M"
    assert config.MAX_TRAINING_STEPS == 100


def test_config_evaluator():
    """Test evaluator configuration."""
    config = GIRAFConfig()
    
    evaluator = config.DEFAULT_EVALUATOR
    
    assert evaluator["steps"] == 150
    assert evaluator["dt_req"] == 1.0
    assert evaluator["gamma"] == 25.0
    assert evaluator["beta"] == 8.0
    assert evaluator["delta"] == 20.0


def test_lora_config():
    """Test LoRA configuration."""
    config = GIRAFConfig()
    
    lora = config.LORA_CONFIG
    
    assert lora["r"] == 8
    assert lora["lora_alpha"] == 32
    assert lora["lora_dropout"] == 0.1
