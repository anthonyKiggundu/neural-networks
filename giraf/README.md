# GIRAF: Governance-Integrated Risk and Assurance Framework

A comprehensive framework for 6G network KPI decision-making using LLM agents with governance-as-code (GaC) risk assessment.

## 📋 Overview

GIRAF provides a sophisticated system for:
- **Real-time Risk Assessment**: Dynamic evaluation of network KPIs with epistemic, environmental, and staleness risk components
- **LLM-based Decision Making**: Fine-tuned language models for network management decisions
- **Trust Calibration**: Confidence alignment between model predictions and ground truth
- **Fraud Detection**: Adversarial behavior identification in 6G networks
- **Comprehensive Visualization**: Advanced plotting for risk analysis and calibration

## 🏗️ Architecture

```
giraf/
├── agents/          # LLM KPI decision agents
├── data/            # Data loading and streaming
├── evaluation/      # Risk calculation and metrics
├── training/        # Model fine-tuning (LoRA)
├── visualization/   # Plotting and diagrams
├── utils/           # Helper utilities
└── config.py        # Configuration management
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/anthonyKiggundu/neural-networks.git
cd neural-networks
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install GIRAF package**:
```bash
pip install -e .
```

### Data Preparation

Place your `cellular_dataframe.parquet` file in the project root directory. The dataset should contain:
- Network KPIs: `ping_ms`, `jitter`, `datarate`, `target_datarate`
- Signal metrics: `PCell_RSRP_1`, `PCell_RSRQ_1`, `PCell_SNR_1`
- Mobility data: `speed_kmh`, `Latitude`, `Longitude`, `Altitude`
- Congestion: `Traffic Jam Factor`
- Device info: `device`, `operator`, `measured_qos`

### Running the Simulation

```bash
python examples/run_simulation.py
```

Or use the command-line interface:
```bash
giraf-simulate
```

## 📚 Usage Examples

### Basic Agent Initialization

```python
from giraf import LLMKPIAgent, GIRAFConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

config = GIRAFConfig()
tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_NAME)

agent = LLMKPIAgent(
    name="Agent-1",
    drift_tag="baseline",
    tokenizer=tokenizer,
    model=model,
    device="cuda"
)
```

### Risk Calculation

```python
from giraf.evaluation import calculate_risk_factors

risk_data = calculate_risk_factors(
    metadata=kpis,
    evaluator=config.DEFAULT_EVALUATOR,
    step=current_step,
    fraud_detected=False,
    behavior_flagged=False,
    threshold=45.0
)

print(f"Aggregate Risk: {risk_data['aggregate_risk']:.2f}")
print(f"Mitigation Signal: {risk_data['mitigation_signal']}")
```

### Custom Fine-tuning

```python
from giraf.training import fine_tune_model
from giraf.data import prepare_dataset

train_data, val_data, test_data = prepare_dataset(data)

model_path = fine_tune_model(
    train_data=train_data,
    base_model_name="EleutherAI/gpt-neo-125M",
    tokenizer=tokenizer,
    val_data=val_data,
    max_steps=100
)
```

### Visualization

```python
from giraf.visualization import (
    extended_visualize_results,
    generate_comparative_reliability_diagram
)

# Generate comprehensive risk plots
extended_visualize_results(
    time_span=time_array,
    aggregate_risks=risk_history,
    epistemic_risks=epi_history,
    staleness_risks=stal_history,
    congestion_index=congestion,
    jitter=jitter_values,
    bt_true=ground_truth,
    bt_reported=reported_conf,
    ping_violations=violations,
    jitter_violations=jitter_viols,
    fraud_detected=fraud_flags
)

# Comparative reliability diagram
generate_comparative_reliability_diagram(
    y_true=actual_outcomes,
    giraf_preds=giraf_predictions,
    pretrained_preds=baseline_predictions
)
```

## 🎛️ Configuration

Edit `giraf/config.py` to customize:

```python
class GIRAFConfig:
    # Risk Model Parameters
    DEFAULT_EVALUATOR = {
        "steps": 150,
        "dt_req": 1.0,      # Latency Deadline (ms)
        "gamma": 25.0,      # Epistemic Risk Weight
        "beta": 8.0,        # Environmental Risk Weight
        "delta": 20.0,      # Staleness Risk Weight
    }
    
    # Thresholds
    MITIGATION_THRESHOLD = 45.0
    TRUST_THRESHOLD = 0.15
    
    # Model Parameters
    BASE_MODEL_NAME = "EleutherAI/gpt-neo-125M"
    MAX_TRAINING_STEPS = 100
```

## 📊 Output Files

The simulation generates:
- `GIRAF_Simulation_Results.pdf`: Comprehensive 6-subplot risk analysis
- `GIRAF_Smoothed_Reliability.pdf`: Calibration comparison
- `risk_distribution_combined.pdf`: Risk by congestion level
- `verified_latency_vs_SMTdepth.png`: Staleness distribution

## 🧪 Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# With coverage
pytest --cov=giraf tests/
```

## 📖 Key Concepts

### Risk Components

1. **Epistemic Risk (rₑₚᵢ)**: Uncertainty from model confidence gaps
   ```
   rₑₚᵢ = γ × (1 - Bₜ)
   ```

2. **Environmental Risk (rₑₙᵥ)**: Network volatility
   ```
   rₑₙᵥ = β × ω²
   ```

3. **Staleness Risk (rₛₜₐₗ)**: Latency penalty
   ```
   rₛₜₐₗ = δ × max(0, Lᵥ - Δtᵣₑq)
   ```

### Trust Calibration

Ground truth confidence is dynamically calculated:
```python
Bₜ = exp(-λ × (σⱼᵢₜₜₑᵣ / SNR))
```

### Mitigation Signal

Binary governance trigger:
```python
signal = 1 if R_total > threshold else 0
```

## 🛠️ System Requirements

- **Python**: 3.8+
- **GPU**: CUDA-capable (recommended for training)
- **RAM**: 16GB minimum (32GB recommended)
- **Disk**: 10GB for models and data

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on Hugging Face Transformers
- Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Inspired by 6G network research and governance-as-code principles

## 📧 Contact

Anthony Kiggundu - [@anthonyKiggundu](https://github.com/anthonyKiggundu)

Project Link: [https://github.com/anthonyKiggundu/neural-networks](https://github.com/anthonyKiggundu/neural-networks/giraf)

## 🔗 References

- Paper: [Your research paper link]
- Documentation: [Full API documentation]
- Issues: [Bug reports and feature requests](https://github.com/anthonyKiggundu/neural-networks/giraf/issues)

---

**Note**: This is research software. Performance may vary based on your dataset and hardware configuration.
