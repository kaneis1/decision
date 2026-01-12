"""
Example script demonstrating how to use SHAP with Decision Transformer models.

This script shows:
1. Loading a trained model
2. Preparing data for SHAP explanation
3. Explaining model predictions
4. Visualizing SHAP values
"""

import numpy as np
import torch
from pathlib import Path
import sys

# Add parent directory to path to import decision_transformer modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization will be disabled.")

try:
    from decision_transformer.explainability.shap_explainer import (
        explain_decision_transformer,
        explain_single_prediction,
        visualize_shap_values,
        ModelWrapperForSHAP,
    )
    SHAP_AVAILABLE = True
except ImportError as e:
    SHAP_AVAILABLE = False
    print(f"Warning: SHAP not available: {e}")
    print("Please install SHAP to use explainability features.")
    print("See INSTALL.md for installation instructions.")

from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.gpt2_bc import GPT2BCModel


def example_basic_shap_usage():
    """
    Basic example: Explain predictions using SHAP.
    """
    print("=" * 60)
    print("Example 1: Basic SHAP Usage")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = 17  # Example: adjust to your state dimension
    act_dim = 1     # Example: adjust to your action dimension
    hidden_size = 128
    max_length = 20
    
    # Load your trained model (replace with actual loading code)
    # model = DecisionTransformer(
    #     state_dim=state_dim,
    #     act_dim=act_dim,
    #     hidden_size=hidden_size,
    #     max_length=max_length,
    # )
    # model.load_state_dict(torch.load("path/to/model.pt"))
    # model = model.to(device)
    
    # For demonstration, create a dummy model
    print("Creating dummy model for demonstration...")
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=hidden_size,
        max_length=max_length,
    ).to(device)
    
    # Generate dummy data
    batch_size = 10
    seq_length = max_length
    
    states = np.random.randn(batch_size, seq_length, state_dim).astype(np.float32)
    actions = np.random.randn(batch_size, seq_length, act_dim).astype(np.float32)
    returns_to_go = np.random.randn(batch_size, seq_length, 1).astype(np.float32)
    timesteps = np.arange(seq_length, dtype=np.int64)
    timesteps = np.tile(timesteps, (batch_size, 1))
    
    # Normalization parameters (if used during training)
    state_mean = np.zeros(state_dim)
    state_std = np.ones(state_dim)
    
    # Feature names (optional, for better visualization)
    feature_names = [f"state_feature_{i}" for i in range(state_dim)]
    
    print(f"\nExplaining {batch_size} trajectories...")
    
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
        background_size=5,
        explainer_type="partition",
        feature_names=feature_names,
    )
    
    print(f"\nSHAP values shape: {results['shap_values'].shape}")
    print(f"Test states shape: {results['test_states'].shape}")
    
    # Visualize
    if MATPLOTLIB_AVAILABLE:
        print("\nGenerating SHAP summary plot...")
        visualize_shap_values(
            results['explanation'],
            plot_type="summary",
            max_display=10,
        )
    else:
        print("\nSkipping visualization (matplotlib not available)")


def example_single_prediction_explanation():
    """
    Example: Explain a single prediction (one trajectory prefix).
    Useful for debugging why the model made a specific decision.
    """
    print("\n" + "=" * 60)
    print("Example 2: Explaining Single Prediction")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = 17
    act_dim = 1
    hidden_size = 128
    max_length = 20
    
    # Create dummy model
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=hidden_size,
        max_length=max_length,
    ).to(device)
    
    # Single trajectory prefix
    seq_length = 10
    states = np.random.randn(seq_length, state_dim).astype(np.float32)
    actions = np.random.randn(seq_length, act_dim).astype(np.float32)
    returns_to_go = np.random.randn(seq_length, 1).astype(np.float32)
    timesteps = np.arange(seq_length, dtype=np.int64)
    
    # Background data
    background_batch_size = 20
    background_states = np.random.randn(
        background_batch_size, max_length, state_dim
    ).astype(np.float32)
    
    state_mean = np.zeros(state_dim)
    state_std = np.ones(state_dim)
    feature_names = [f"feature_{i}" for i in range(state_dim)]
    
    print(f"\nExplaining single prediction for trajectory of length {seq_length}...")
    
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
        feature_names=feature_names,
    )
    
    print(f"\nPredicted action: {results['predicted_action']}")
    print(f"SHAP values shape: {results['shap_values'].shape}")
    print(f"\nTop contributing features (by absolute SHAP value):")
    
    # Get mean absolute SHAP values per feature
    mean_shap_per_feature = np.abs(results['shap_values']).mean(axis=0)
    top_features = np.argsort(mean_shap_per_feature)[::-1][:5]
    
    for idx in top_features:
        print(f"  {feature_names[idx]}: {mean_shap_per_feature[idx]:.4f}")


def example_gpt2_bc_model():
    """
    Example: Using SHAP with GPT2BCModel.
    """
    print("\n" + "=" * 60)
    print("Example 3: SHAP with GPT2BCModel")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = 17
    act_dim = 1
    hidden_size = 128
    n_layer = 4
    n_head = 4
    max_length = 20
    
    # Create GPT2BCModel
    model = GPT2BCModel(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=hidden_size,
        n_layer=n_layer,
        n_head=n_head,
        max_length=max_length,
    ).to(device)
    
    # Generate dummy data
    batch_size = 10
    seq_length = max_length
    
    states = np.random.randn(batch_size, seq_length, state_dim).astype(np.float32)
    actions = np.zeros((batch_size, seq_length, act_dim), dtype=np.float32)
    returns_to_go = np.zeros((batch_size, seq_length, 1), dtype=np.float32)
    timesteps = np.arange(seq_length, dtype=np.int64)
    timesteps = np.tile(timesteps, (batch_size, 1))
    
    state_mean = np.zeros(state_dim)
    state_std = np.ones(state_dim)
    
    print(f"\nExplaining GPT2BCModel predictions...")
    
    # The same explain_decision_transformer function works with GPT2BCModel
    # because it handles different model signatures automatically
    results = explain_decision_transformer(
        model=model,
        states=states,
        actions=actions,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        device=device,
        state_mean=state_mean,
        state_std=state_std,
        background_size=5,
        explainer_type="partition",
    )
    
    print(f"SHAP values computed successfully!")
    print(f"SHAP values shape: {results['shap_values'].shape}")


def example_custom_wrapper():
    """
    Example: Using ModelWrapperForSHAP directly for more control.
    """
    print("\n" + "=" * 60)
    print("Example 4: Using Custom Wrapper")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = 17
    act_dim = 1
    hidden_size = 128
    max_length = 20
    
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=hidden_size,
        max_length=max_length,
    ).to(device)
    
    # Create wrapper
    wrapper = ModelWrapperForSHAP(
        model=model,
        device=device,
        state_mean=np.zeros(state_dim),
        state_std=np.ones(state_dim),
    )
    
    # Create data for SHAP
    if not SHAP_AVAILABLE:
        print("SHAP not available, skipping this example")
        return
    import shap
    
    batch_size = 1
    background_data = np.random.randn(5, max_length * state_dim).astype(np.float32)
    test_data = np.random.randn(1, max_length * state_dim).astype(np.float32)
    
    print("\nUsing PartitionExplainer with custom wrapper...")
    
    explainer = shap.PartitionExplainer(wrapper, background_data)
    shap_values = explainer(test_data)
    
    print(f"SHAP values computed!")
    print(f"SHAP values shape: {shap_values.values.shape}")
    
    # Visualize
    if MATPLOTLIB_AVAILABLE:
        shap.plots.bar(shap_values[0], show=False)
        plt.title("SHAP Values for Action Prediction")
        plt.tight_layout()
        plt.show()
    else:
        print("Skipping visualization (matplotlib not available)")


def example_time_step_importance():
    """
    Example: Analyzing which timesteps are most important for the prediction.
    """
    print("\n" + "=" * 60)
    print("Example 5: Timestep Importance Analysis")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = 17
    act_dim = 1
    hidden_size = 128
    max_length = 20
    
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=hidden_size,
        max_length=max_length,
    ).to(device)
    
    seq_length = 15
    states = np.random.randn(seq_length, state_dim).astype(np.float32)
    actions = np.random.randn(seq_length, act_dim).astype(np.float32)
    returns_to_go = np.random.randn(seq_length, 1).astype(np.float32)
    timesteps = np.arange(seq_length, dtype=np.int64)
    
    background_states = np.random.randn(20, max_length, state_dim).astype(np.float32)
    
    print(f"\nAnalyzing timestep importance for trajectory of length {seq_length}...")
    
    results = explain_single_prediction(
        model=model,
        states=states,
        actions=actions,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        device=device,
        background_states=background_states,
    )
    
    # Analyze SHAP values per timestep
    shap_values = results['shap_values']  # (seq_length, state_dim)
    
    # Sum absolute SHAP values per timestep
    importance_per_timestep = np.abs(shap_values).sum(axis=1)
    
    print("\nTimestep importance (higher = more important):")
    for t, importance in enumerate(importance_per_timestep):
        print(f"  Timestep {t:2d}: {importance:.4f}")
    
    # Visualize
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(10, 6))
        plt.plot(importance_per_timestep, marker='o')
        plt.xlabel("Timestep")
        plt.ylabel("Total Absolute SHAP Value")
        plt.title("Timestep Importance for Action Prediction")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("Skipping visualization (matplotlib not available)")


if __name__ == "__main__":
    print("SHAP Explanation Examples for Decision Transformer Models")
    print("=" * 60)
    
    if not SHAP_AVAILABLE:
        print("\nERROR: SHAP is not installed!")
        print("Please install SHAP using one of these methods:")
        print("  1. conda install -c conda-forge shap")
        print("  2. pip install shap (if you have network access)")
        print("  3. See INSTALL.md for network-restricted installation options")
        sys.exit(1)
    
    # Run examples
    try:
        example_basic_shap_usage()
    except Exception as e:
        print(f"Error in example 1: {e}")
    
    try:
        example_single_prediction_explanation()
    except Exception as e:
        print(f"Error in example 2: {e}")
    
    try:
        example_gpt2_bc_model()
    except Exception as e:
        print(f"Error in example 3: {e}")
    
    try:
        example_custom_wrapper()
    except Exception as e:
        print(f"Error in example 4: {e}")
    
    try:
        example_time_step_importance()
    except Exception as e:
        print(f"Error in example 5: {e}")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)

