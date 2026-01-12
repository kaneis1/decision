"""
SHAP (SHapley Additive exPlanations) integration for Decision Transformer models.

This module provides utilities to explain model predictions using SHAP.
Supports DecisionTransformer, GPT2BCModel, and other trajectory models.
"""

import numpy as np
import torch
import inspect
from typing import Dict, List, Tuple, Optional, Union

# Make SHAP and matplotlib optional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


class ModelWrapperForSHAP:
    """
    Wrapper class to make Decision Transformer models compatible with SHAP explainers.
    
    SHAP expects models that take numpy arrays and return numpy arrays.
    This wrapper handles the conversion between numpy and torch tensors,
    and manages the complex input structure of Decision Transformer models.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        explain_features: str = "states",  # "states", "returns_to_go", "both"
        baseline_states: Optional[np.ndarray] = None,
        baseline_actions: Optional[np.ndarray] = None,
        baseline_returns_to_go: Optional[np.ndarray] = None,
    ):
        """
        Args:
            model: The Decision Transformer model (DecisionTransformer, GPT2BCModel, etc.)
            device: torch.device to run the model on
            state_mean: Mean for state normalization (if used during training)
            state_std: Std for state normalization (if used during training)
            explain_features: Which features to explain ("states", "returns_to_go", "both")
            baseline_states: Baseline states for SHAP (default: zeros)
            baseline_actions: Baseline actions for SHAP (default: zeros)
            baseline_returns_to_go: Baseline returns-to-go for SHAP (default: zeros)
        """
        self.model = model
        self.device = device
        self.model.eval()  # Set to evaluation mode
        
        self.state_mean = state_mean if state_mean is not None else 0.0
        self.state_std = state_std if state_std is not None else 1.0
        self.explain_features = explain_features
        
        # Get model forward signature to handle different model types
        forward_sig = inspect.signature(model.forward)
        self.forward_params = list(forward_sig.parameters.keys())
        
        # Store baseline values
        self.baseline_states = baseline_states
        self.baseline_actions = baseline_actions
        self.baseline_returns_to_go = baseline_returns_to_go
        
        # Cache model parameters
        if hasattr(model, 'state_dim'):
            self.state_dim = model.state_dim
        if hasattr(model, 'act_dim'):
            self.act_dim = model.act_dim
        if hasattr(model, 'max_length'):
            self.max_length = model.max_length
    
    def __call__(self, states_array: np.ndarray) -> np.ndarray:
        """
        Forward pass for SHAP explainer.
        
        Args:
            states_array: Flattened states array (will be reshaped)
            
        Returns:
            Flattened action predictions
        """
        # Reshape input - SHAP passes flattened arrays
        # Assuming states_array is (batch_size * seq_length * state_dim,)
        # We need to reshape it back to (batch_size, seq_length, state_dim)
        
        batch_size = states_array.shape[0]
        seq_length = self.max_length if self.max_length else states_array.shape[0]
        state_dim = self.state_dim
        
        # Reshape states
        states = states_array.reshape(batch_size, seq_length, state_dim)
        
        # Normalize states
        states = (states - self.state_mean) / (self.state_std + 1e-8)
        
        # Convert to torch tensors
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        
        # Prepare other inputs (use baselines or zeros)
        batch_size_actual, seq_length_actual = states_t.shape[:2]
        
        if self.baseline_actions is not None:
            actions_t = torch.as_tensor(
                self.baseline_actions[:batch_size_actual, :seq_length_actual],
                dtype=torch.float32, device=self.device
            )
        else:
            actions_t = torch.zeros(
                (batch_size_actual, seq_length_actual, self.act_dim),
                dtype=torch.float32, device=self.device
            )
        
        if self.baseline_returns_to_go is not None:
            returns_to_go_t = torch.as_tensor(
                self.baseline_returns_to_go[:batch_size_actual, :seq_length_actual],
                dtype=torch.float32, device=self.device
            )
        else:
            returns_to_go_t = torch.zeros(
                (batch_size_actual, seq_length_actual, 1),
                dtype=torch.float32, device=self.device
            )
        
        timesteps_t = torch.arange(seq_length_actual, dtype=torch.long, device=self.device)
        timesteps_t = timesteps_t.unsqueeze(0).expand(batch_size_actual, -1)
        
        attention_mask = torch.ones(
            (batch_size_actual, seq_length_actual),
            dtype=torch.long, device=self.device
        )
        
        # Call model forward with appropriate signature
        with torch.no_grad():
            if 'mo' in self.forward_params:
                # DecisionTransformer with MO support
                _, action_preds, _ = self.model.forward(
                    states_t, actions_t, None,
                    returns_to_go_t[:, :-1] if returns_to_go_t.shape[1] > 1 else returns_to_go_t,
                    timesteps_t, attention_mask=attention_mask, mo=None
                )
            elif 'returns_to_go' in self.forward_params:
                # DecisionTransformer without MO
                _, action_preds, _ = self.model.forward(
                    states_t, actions_t, None,
                    returns_to_go_t[:, :-1] if returns_to_go_t.shape[1] > 1 else returns_to_go_t,
                    timesteps_t, attention_mask=attention_mask
                )
            elif 'timesteps' in self.forward_params:
                # GPT2BCModel
                _, action_preds, _ = self.model.forward(
                    states_t, actions_t, None,
                    timesteps=timesteps_t, attention_mask=attention_mask
                )
            else:
                # Fallback
                _, action_preds, _ = self.model.forward(
                    states_t, actions_t, None,
                    returns_to_go_t[:, :-1] if returns_to_go_t.shape[1] > 1 else returns_to_go_t,
                    timesteps_t, attention_mask=attention_mask
                )
        
        # Handle different output shapes
        if action_preds.shape[1] == 1:
            # GPT2BCModel: (B, 1, act_dim) - use last action
            actions = action_preds[:, -1, :].cpu().numpy()
        else:
            # DecisionTransformer: (B, T, act_dim) - use last timestep action
            actions = action_preds[:, -1, :].cpu().numpy()
        
        return actions.flatten()


def _check_shap_available():
    """Check if SHAP is available and raise informative error if not."""
    if not SHAP_AVAILABLE:
        raise ImportError(
            "SHAP is not installed. Please install it using one of these methods:\n"
            "1. conda install -c conda-forge shap\n"
            "2. pip install shap (if you have network access)\n"
            "3. pip install --index-url <local-index> shap (if using a local package index)"
        )


def explain_with_deep_explainer(
    model: torch.nn.Module,
    background_data: np.ndarray,
    test_data: np.ndarray,
    device: torch.device,
    state_mean: Optional[np.ndarray] = None,
    state_std: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
) -> shap.Explanation:
    """
    Explain model predictions using SHAP DeepExplainer.
    
    DeepExplainer uses DeepLIFT algorithm and is optimized for deep learning models.
    
    Args:
        model: The Decision Transformer model
        background_data: Background dataset for SHAP (shape: batch_size, seq_length, state_dim)
        test_data: Test samples to explain (shape: batch_size, seq_length, state_dim)
        device: torch.device
        state_mean: Mean for state normalization
        state_std: Std for state normalization
        feature_names: Names for each feature dimension
        
    Returns:
        SHAP Explanation object
    """
    _check_shap_available()
    
    # Normalize background and test data
    if state_mean is not None and state_std is not None:
        background_data = (background_data - state_mean) / (state_std + 1e-8)
        test_data = (test_data - state_mean) / (state_std + 1e-8)
    
    # Convert to torch tensors
    background_t = torch.as_tensor(background_data, dtype=torch.float32, device=device)
    test_t = torch.as_tensor(test_data, dtype=torch.float32, device=device)
    
    model.eval()
    
    # Create explainer
    explainer = shap.DeepExplainer(model, background_t)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(test_t)
    
    # Create explanation object
    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=test_data,
        feature_names=feature_names
    )
    
    return explanation


def explain_with_gradient_explainer(
    model: torch.nn.Module,
    background_data: np.ndarray,
    test_data: np.ndarray,
    device: torch.device,
    state_mean: Optional[np.ndarray] = None,
    state_std: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
) -> shap.Explanation:
    """
    Explain model predictions using SHAP GradientExplainer.
    
    GradientExplainer uses gradients and is good for models where gradients are available.
    
    Args:
        model: The Decision Transformer model
        background_data: Background dataset for SHAP
        test_data: Test samples to explain
        device: torch.device
        state_mean: Mean for state normalization
        state_std: Std for state normalization
        feature_names: Names for each feature dimension
        
    Returns:
        SHAP Explanation object
    """
    _check_shap_available()
    
    # Normalize data
    if state_mean is not None and state_std is not None:
        background_data = (background_data - state_mean) / (state_std + 1e-8)
        test_data = (test_data - state_mean) / (state_std + 1e-8)
    
    # Convert to torch tensors
    background_t = torch.as_tensor(background_data, dtype=torch.float32, device=device)
    test_t = torch.as_tensor(test_data, dtype=torch.float32, device=device)
    
    model.eval()
    
    # Create explainer
    explainer = shap.GradientExplainer(model, background_t)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(test_t)
    
    # Create explanation object
    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=test_data,
        feature_names=feature_names
    )
    
    return explanation


def explain_decision_transformer(
    model: torch.nn.Module,
    states: np.ndarray,
    actions: np.ndarray,
    returns_to_go: np.ndarray,
    timesteps: np.ndarray,
    device: torch.device,
    state_mean: Optional[np.ndarray] = None,
    state_std: Optional[np.ndarray] = None,
    background_size: int = 50,
    explainer_type: str = "deep",  # "deep", "gradient", "partition"
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """
    High-level function to explain Decision Transformer predictions.
    
    Args:
        model: Decision Transformer model
        states: States to explain (batch_size, seq_length, state_dim)
        actions: Actions (batch_size, seq_length, act_dim)
        returns_to_go: Returns-to-go (batch_size, seq_length, 1)
        timesteps: Timesteps (batch_size, seq_length)
        device: torch.device
        state_mean: Mean for state normalization
        state_std: Std for state normalization
        background_size: Number of background samples for SHAP
        explainer_type: Type of SHAP explainer
        feature_names: Names for state features
        
    Returns:
        Dictionary with SHAP values and visualizations
    """
    _check_shap_available()
    model.eval()
    
    # Sample background data
    if background_size > len(states):
        background_size = len(states)
    background_indices = np.random.choice(len(states), background_size, replace=False)
    background_states = states[background_indices]
    
    # Use first few samples as test data (or all if small)
    test_size = min(5, len(states))
    test_states = states[:test_size]
    
    # Create wrapper for model
    wrapper = ModelWrapperForSHAP(
        model=model,
        device=device,
        state_mean=state_mean,
        state_std=state_std,
        baseline_actions=actions[background_indices],
        baseline_returns_to_go=returns_to_go[background_indices],
    )
    
    # Flatten data for SHAP (batch_size * seq_length * state_dim)
    background_flat = background_states.reshape(-1, background_states.shape[-1])
    test_flat = test_states.reshape(-1, test_states.shape[-1])
    
    if explainer_type == "partition":
        # Use PartitionExplainer which works with any model
        _check_shap_available()
        explainer = shap.PartitionExplainer(wrapper, background_flat)
        shap_values_flat = explainer(test_flat)
        
        # Reshape back to original dimensions
        shap_values = shap_values_flat.values.reshape(
            test_states.shape[0], test_states.shape[1], test_states.shape[2]
        )
        
        explanation = shap.Explanation(
            values=shap_values,
            base_values=shap_values_flat.base_values,
            data=test_states,
            feature_names=feature_names
        )
    else:
        # For DeepExplainer or GradientExplainer, we need to handle the model differently
        # These work better when explaining individual timesteps
        print(f"Note: {explainer_type} explainer may require custom implementation for sequential models.")
        print("Using PartitionExplainer instead...")
        _check_shap_available()
        explainer = shap.PartitionExplainer(wrapper, background_flat)
        shap_values_flat = explainer(test_flat)
        
        shap_values = shap_values_flat.values.reshape(
            test_states.shape[0], test_states.shape[1], test_states.shape[2]
        )
        
        explanation = shap.Explanation(
            values=shap_values,
            base_values=shap_values_flat.base_values,
            data=test_states,
            feature_names=feature_names
        )
    
    return {
        "explanation": explanation,
        "shap_values": shap_values,
        "test_states": test_states,
    }


def visualize_shap_values(
    explanation: shap.Explanation,
    save_path: Optional[str] = None,
    plot_type: str = "summary",
    max_display: int = 20,
):
    """
    Visualize SHAP values.
    
    Args:
        explanation: SHAP Explanation object
        save_path: Path to save the plot (optional)
        plot_type: Type of plot ("summary", "waterfall", "bar")
        max_display: Maximum number of features to display
    """
    _check_shap_available()
    
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is not installed. Please install it using:\n"
            "conda install matplotlib or pip install matplotlib"
        )
    
    if plot_type == "summary":
        shap.summary_plot(explanation, show=False, max_display=max_display)
    elif plot_type == "waterfall":
        shap.waterfall_plot(explanation[0], show=False)
    elif plot_type == "bar":
        shap.plots.bar(explanation, show=False, max_display=max_display)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def explain_single_prediction(
    model: torch.nn.Module,
    states: np.ndarray,  # (seq_length, state_dim)
    actions: np.ndarray,  # (seq_length, act_dim)
    returns_to_go: np.ndarray,  # (seq_length, 1)
    timesteps: np.ndarray,  # (seq_length,)
    device: torch.device,
    background_states: np.ndarray,
    state_mean: Optional[np.ndarray] = None,
    state_std: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """
    Explain a single prediction (one trajectory prefix).
    
    This is useful for understanding why the model made a specific action prediction.
    
    Args:
        model: Decision Transformer model
        states: Single trajectory states (seq_length, state_dim)
        actions: Single trajectory actions (seq_length, act_dim)
        returns_to_go: Single trajectory returns (seq_length, 1)
        timesteps: Single trajectory timesteps (seq_length,)
        device: torch.device
        background_states: Background states for SHAP (batch_size, seq_length, state_dim)
        state_mean: Mean for normalization
        state_std: Std for normalization
        feature_names: Feature names
        
    Returns:
        Dictionary with SHAP values and predictions
    """
    _check_shap_available()
    
    # Reshape to batch dimension
    states_batch = states[np.newaxis, :, :]  # (1, seq_length, state_dim)
    actions_batch = actions[np.newaxis, :, :]  # (1, seq_length, act_dim)
    returns_to_go_batch = returns_to_go[np.newaxis, :, :]  # (1, seq_length, 1)
    timesteps_batch = timesteps[np.newaxis, :]  # (1, seq_length)
    
    # Create wrapper
    wrapper = ModelWrapperForSHAP(
        model=model,
        device=device,
        state_mean=state_mean,
        state_std=state_std,
        baseline_actions=actions_batch,
        baseline_returns_to_go=returns_to_go_batch,
    )
    
    # Flatten for SHAP
    states_flat = states.flatten()  # (seq_length * state_dim,)
    background_flat = background_states.reshape(
        background_states.shape[0] * background_states.shape[1], -1
    )
    
    # Use PartitionExplainer
    _check_shap_available()
    explainer = shap.PartitionExplainer(wrapper, background_flat)
    shap_values_flat = explainer(states_flat)
    
    # Reshape back
    shap_values = shap_values_flat.values.reshape(states.shape[0], states.shape[1])
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        states_t = torch.as_tensor(states_batch, dtype=torch.float32, device=device)
        if state_mean is not None and state_std is not None:
            states_t = (states_t - torch.as_tensor(state_mean, device=device)) / (
                torch.as_tensor(state_std, device=device) + 1e-8
            )
        actions_t = torch.as_tensor(actions_batch, dtype=torch.float32, device=device)
        returns_to_go_t = torch.as_tensor(returns_to_go_batch, dtype=torch.float32, device=device)
        timesteps_t = torch.as_tensor(timesteps_batch, dtype=torch.long, device=device)
        
        forward_sig = inspect.signature(model.forward)
        forward_params = list(forward_sig.parameters.keys())
        
        if 'returns_to_go' in forward_params:
            _, action_pred, _ = model.forward(
                states_t, actions_t, None,
                returns_to_go_t[:, :-1], timesteps_t,
                attention_mask=torch.ones((1, states_t.shape[1]), device=device)
            )
        else:
            _, action_pred, _ = model.forward(
                states_t, actions_t, None,
                timesteps=timesteps_t,
                attention_mask=torch.ones((1, states_t.shape[1]), device=device)
            )
        
        predicted_action = action_pred[0, -1].cpu().numpy()
    
    return {
        "shap_values": shap_values,
        "predicted_action": predicted_action,
        "states": states,
        "explanation": shap_values_flat,
    }

