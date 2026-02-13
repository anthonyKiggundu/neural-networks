"""
Configuration settings for GIRAF framework.
"""

class GIRAFConfig:
    """Central configuration for GIRAF system."""
    
    # Simulation Parameters
    EPOCH_DURATION_MS = 100  # Each decision epoch duration
    
    # Risk Model Parameters
    DEFAULT_EVALUATOR = {
        "steps": 150,
        "dt_req": 1.0,      # Latency Deadline (ms)
        "p_base": 10.0,     # Baseline Premium ($)
        "gamma": 25.0,      # Epistemic Risk Weight
        "beta": 8.0,        # Environmental Risk Weight
        "delta": 20.0,      # Staleness Risk Weight
    }
    
    # Thresholds
    MITIGATION_THRESHOLD = 45.0
    TRUST_THRESHOLD = 0.15
    LAMBDA_SENSITIVITY = 0.25
    
    # Model Parameters
    BASE_MODEL_NAME = "EleutherAI/gpt-neo-125M"
    MAX_TRAINING_STEPS = 100
    LEARNING_RATE = 2e-4
    
    # LoRA Configuration
    LORA_CONFIG = {
        "r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
    }
    
    # Data Split Ratios
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2
    
    # SLA Parameters
    SLA_PING_PERCENTILE = 95
    SLA_JITTER_PERCENTILE = 95
    DEFAULT_SLA_PING = 30.0
    DEFAULT_SLA_JITTER = 10.0
    
    # Visualization
    N_BINS = 10
    PLOT_DPI = 300
