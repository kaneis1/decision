# SHAP Integration for Decision Transformer Models

This directory contains utilities for explaining Decision Transformer model predictions using SHAP (SHapley Additive exPlanations).

## Installation

First, install SHAP. If you're on a compute cluster without direct internet access, see [INSTALL.md](INSTALL.md) for detailed installation instructions.

### Quick Install (if you have network access)

```bash
pip install shap
```

### Recommended: Install via Conda

```bash
conda install -c conda-forge shap
```

### Network-Restricted Environments

If `pip install shap` fails due to network issues, try:

1. **Conda (recommended)**: `conda install -c conda-forge shap`
2. **Local package index**: See INSTALL.md
3. **Manual download**: See INSTALL.md for step-by-step instructions

The code will work without SHAP installed - you'll only get errors when actually trying to use the explainability functions.

## Quick Start

### Do I Need to Save the Model First?

**Short answer: It depends!**

- **No, you don't need to save** if the model is already in memory (e.g., you just finished training)
- **Yes, you should save** if you want to explain it later or in a different session

See `load_model_for_shap.py` for examples of both scenarios.

### Basic Usage

```python
from decision_transformer.explainability.shap_explainer import explain_decision_transformer
from decision_transformer.explainability.load_model_for_shap import load_decision_transformer_model
import numpy as np
import torch

# Option 1: Load a saved model
model = load_decision_transformer_model(
    model_path="models/my_model.pt",
    state_dim=17,
    act_dim=1,
    hidden_size=128,
    max_length=20,
)

# Option 2: Use model already in memory (e.g., after training)
# model = your_trained_model  # Already in memory
# model.eval()  # Make sure it's in eval mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare your data
states = np.array(...)  # shape: (batch_size, seq_length, state_dim)
actions = np.array(...)  # shape: (batch_size, seq_length, act_dim)
returns_to_go = np.array(...)  # shape: (batch_size, seq_length, 1)
timesteps = np.array(...)  # shape: (batch_size, seq_length)

# Normalization parameters (if used during training)
state_mean = np.array(...)
state_std = np.array(...)

# Explain predictions
results = explain_decision_transformer(
    model=model,
    states=states,
    actions=actions,
    returns_to_go=returns_to_go,
    timesteps=timesteps,
    device=device,
    state_mean=state_mean,
    state_std=state_std,
    background_size=50,
)

# Visualize
from decision_transformer.explainability.shap_explainer import visualize_shap_values
visualize_shap_values(results['explanation'], plot_type="summary")
```

### Explain Single Prediction

```python
from decision_transformer.explainability.shap_explainer import explain_single_prediction

# Single trajectory prefix
states = np.array(...)  # shape: (seq_length, state_dim)
actions = np.array(...)  # shape: (seq_length, act_dim)
returns_to_go = np.array(...)  # shape: (seq_length, 1)
timesteps = np.array(...)  # shape: (seq_length,)

# Background data
background_states = np.array(...)  # shape: (batch_size, seq_length, state_dim)

results = explain_single_prediction(
    model=model,
    states=states,
    actions=actions,
    returns_to_go=returns_to_go,
    timesteps=timesteps,
    device=device,
    background_states=background_states,
    state_mean=state_mean,
    state_std=state_std,
)

print(f"Predicted action: {results['predicted_action']}")
print(f"SHAP values shape: {results['shap_values'].shape}")
```

## Features

### Supported Models

- `DecisionTransformer`: Standard Decision Transformer model
- `GPT2BCModel`: GPT-2 based behavior cloning model
- `BertDecisionTransformer`: BERT-based Decision Transformer
- Any model inheriting from `TrajectoryModel`

### SHAP Explainers

The integration uses **PartitionExplainer** by default, which works with any model type. It:

- Handles complex input structures (states, actions, returns_to_go, timesteps)
- Supports sequential data (trajectories)
- Works with both continuous and discrete actions
- Automatically detects model type and adapts the forward pass

### What Can Be Explained

1. **State Feature Importance**: Which state features most influence the action prediction
2. **Timestep Importance**: Which timesteps in the trajectory are most important
3. **Interaction Effects**: How different features interact to influence predictions

## API Reference

### `explain_decision_transformer`

High-level function to explain multiple predictions.

**Parameters:**
- `model`: The Decision Transformer model
- `states`: States array (batch_size, seq_length, state_dim)
- `actions`: Actions array (batch_size, seq_length, act_dim)
- `returns_to_go`: Returns-to-go array (batch_size, seq_length, 1)
- `timesteps`: Timesteps array (batch_size, seq_length)
- `device`: torch.device
- `state_mean`: Mean for state normalization (optional)
- `state_std`: Std for state normalization (optional)
- `background_size`: Number of background samples (default: 50)
- `explainer_type`: Type of explainer - "partition" (default), "deep", "gradient"
- `feature_names`: List of feature names for visualization (optional)

**Returns:**
- Dictionary with:
  - `explanation`: SHAP Explanation object
  - `shap_values`: SHAP values array
  - `test_states`: Test states used

### `explain_single_prediction`

Explain a single trajectory prefix prediction.

**Parameters:**
- `model`: The Decision Transformer model
- `states`: Single trajectory states (seq_length, state_dim)
- `actions`: Single trajectory actions (seq_length, act_dim)
- `returns_to_go`: Single trajectory returns (seq_length, 1)
- `timesteps`: Single trajectory timesteps (seq_length,)
- `device`: torch.device
- `background_states`: Background states for SHAP (batch_size, seq_length, state_dim)
- `state_mean`: Mean for normalization (optional)
- `state_std`: Std for normalization (optional)
- `feature_names`: Feature names (optional)

**Returns:**
- Dictionary with:
  - `shap_values`: SHAP values (seq_length, state_dim)
  - `predicted_action`: Predicted action
  - `states`: Input states
  - `explanation`: SHAP Explanation object

### `ModelWrapperForSHAP`

Wrapper class that makes models compatible with SHAP.

**Use case:** When you need more control over the explanation process.

```python
from decision_transformer.explainability.shap_explainer import ModelWrapperForSHAP
import shap

wrapper = ModelWrapperForSHAP(
    model=model,
    device=device,
    state_mean=state_mean,
    state_std=state_std,
)

background_data = ...  # Flattened states
test_data = ...  # Flattened states

explainer = shap.PartitionExplainer(wrapper, background_data)
shap_values = explainer(test_data)
```

## Examples

### Loading Models for SHAP

See `load_model_for_shap.py` for examples of:
- Loading saved model checkpoints
- Using in-memory models (no save/load needed)
- Saving models during training

### SHAP Usage Examples

See `example_shap_usage.py` for complete examples including:

1. Basic SHAP usage
2. Single prediction explanation
3. GPT2BCModel usage
4. Custom wrapper usage
5. Timestep importance analysis

Run examples:

```bash
python -m decision_transformer.explainability.example_shap_usage
```

## Visualization

### Summary Plot

Shows feature importance across multiple samples:

```python
visualize_shap_values(explanation, plot_type="summary", max_display=20)
```

### Waterfall Plot

Shows how each feature contributes to a single prediction:

```python
visualize_shap_values(explanation, plot_type="waterfall")
```

### Bar Plot

Shows mean absolute SHAP values:

```python
visualize_shap_values(explanation, plot_type="bar")
```

### Custom Visualizations

You can also use SHAP's built-in plotting functions directly:

```python
import shap

# Summary plot
shap.summary_plot(explanation)

# Force plot (for single prediction)
shap.force_plot(explainer.expected_value, shap_values[0], test_data[0])

# Waterfall plot
shap.waterfall_plot(explanation[0])
```

## Tips

1. **Saving Models**: If you want to explain models later, save them during training:
   ```python
   torch.save(model.state_dict(), "models/my_model.pt")
   ```
   Or save full checkpoint with metadata:
   ```python
   checkpoint = {
       'model_state_dict': model.state_dict(),
       'state_dim': state_dim,
       'act_dim': act_dim,
       # ... other metadata
   }
   torch.save(checkpoint, "models/my_model.pt")
   ```

2. **Model State**: Always set `model.eval()` before using SHAP to ensure consistent behavior.

3. **Background Data**: Use a representative sample from your training data as background. 50-100 samples usually work well.

2. **Normalization**: Always provide `state_mean` and `state_std` if you normalized during training.

4. **Feature Names**: Provide meaningful feature names for better visualization:
   ```python
   feature_names = ["risk", "error", "delta", "infinity", "continuous", ...]
   ```

5. **Memory**: For large models or long sequences, use smaller `background_size` or explain fewer samples at once.

6. **Timestep Analysis**: To understand which timesteps are most important:
   ```python
   shap_values = results['shap_values']  # (seq_length, state_dim)
   importance_per_timestep = np.abs(shap_values).sum(axis=1)
   ```

## Troubleshooting

### Out of Memory

- Reduce `background_size`
- Use CPU instead of GPU for explanation
- Process fewer samples at a time

### Slow Computation

- PartitionExplainer is slower but works with any model
- Consider using fewer background samples
- Use GPU for model inference

### Unexpected Results

- Ensure normalization matches training
- Check that model is in eval mode
- Verify input shapes match expected format

## References

- [SHAP Documentation](https://shap.readthedocs.io/)
- [SHAP GitHub](https://github.com/slundberg/shap)
- [Decision Transformer Paper](https://arxiv.org/abs/2106.01345)

