"""
SHAP explainer specifically for IPD (Iterated Prisoner's Dilemma) Decision Transformer models.

This module provides utilities to:
1. Load IPD data using load_ipd_csv_as_trajectories
2. Create feature name mappings for SHAP explanations
3. Explain IPD Decision Transformer model predictions

IPD is a binary classification task: defect (0) and coop (1)
"""

import numpy as np
import torch
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import IPD data loading function
from ipd.experiment_ipd import load_ipd_csv_as_trajectories

# Import existing SHAP utilities
try:
    from decision_transformer.explainability.shap_explainer import (
        explain_decision_transformer,
        SHAPexplainerDT,
    )
except ImportError:
    # These should not be required if you only use IPD-specific utilities
    explain_decision_transformer = None
    SHAPexplainerDT = None

try:
    import shap
except ImportError:
    shap = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import rc, font_manager
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None

import inspect


class _IPDSHAPWrapper:
    """Lightweight SHAP wrapper for IPD when shap_explainer.py is absent."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        baseline_actions: np.ndarray,
        baseline_returns_to_go: np.ndarray,
        max_ep_len: Optional[int] = None,
        max_length: Optional[int] = None,
    ):
        self.model = model
        self.device = device
        self.state_mean = state_mean
        self.state_std = state_std
        self.baseline_actions = baseline_actions
        self.baseline_returns_to_go = baseline_returns_to_go
        self.max_ep_len = int(max_ep_len) if max_ep_len is not None else None
        self.max_length = max_length

        self.model.to(self.device)
        self.model.eval()

        forward_sig = inspect.signature(model.forward)
        self.forward_params = list(forward_sig.parameters.keys())

        self.state_dim = getattr(model, "state_dim", baseline_actions.shape[-1])
        self.act_dim = getattr(model, "act_dim", baseline_actions.shape[-1])

    def predict(self, states_array: np.ndarray) -> np.ndarray:
        # states_array: (n_samples, seq_len * state_dim)
        n_samples = states_array.shape[0]
        total_features = states_array.shape[1]

        if self.max_length:
            seq_length = self.max_length
            state_dim = self.state_dim
        else:
            state_dim = self.state_dim
            seq_length = total_features // state_dim
            if total_features % state_dim != 0:
                raise ValueError(
                    f"Cannot reshape {total_features} features into {state_dim}-dimensional states"
                )

        states = states_array.reshape(n_samples, seq_length, state_dim)
        states = (states - self.state_mean) / (self.state_std + 1e-8)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)

        batch_size_actual, seq_length_actual = states_t.shape[:2]

        actions_t = torch.as_tensor(
            self.baseline_actions[:batch_size_actual, :seq_length_actual],
            dtype=torch.float32,
            device=self.device,
        )
        returns_to_go_t = torch.as_tensor(
            self.baseline_returns_to_go[:batch_size_actual, :seq_length_actual],
            dtype=torch.float32,
            device=self.device,
        )

        timesteps_t = torch.arange(seq_length_actual, dtype=torch.long, device=self.device)
        if self.max_ep_len is not None and self.max_ep_len > 0:
            timesteps_t = torch.clamp(timesteps_t, max=self.max_ep_len - 1)
        timesteps_t = timesteps_t.unsqueeze(0).expand(batch_size_actual, -1)

        attention_mask = torch.ones(
            (batch_size_actual, seq_length_actual),
            dtype=torch.long,
            device=self.device,
        )

        # Align returns-to-go length
        if returns_to_go_t.shape[1] == seq_length_actual + 1:
            returns_to_go_aligned = returns_to_go_t[:, :-1]
        elif returns_to_go_t.shape[1] == seq_length_actual:
            returns_to_go_aligned = returns_to_go_t
        else:
            if returns_to_go_t.shape[1] > seq_length_actual:
                returns_to_go_aligned = returns_to_go_t[:, :seq_length_actual]
            else:
                pad_len = seq_length_actual - returns_to_go_t.shape[1]
                pad = torch.zeros(
                    (batch_size_actual, pad_len, 1),
                    dtype=returns_to_go_t.dtype,
                    device=returns_to_go_t.device,
                )
                returns_to_go_aligned = torch.cat([returns_to_go_t, pad], dim=1)

        with torch.no_grad():
            if "returns_to_go" in self.forward_params:
                _, action_preds, _ = self.model.forward(
                    states_t,
                    actions_t,
                    None,
                    returns_to_go_aligned,
                    timesteps_t,
                    attention_mask=attention_mask,
                )
            else:
                _, action_preds, _ = self.model.forward(
                    states_t,
                    actions_t,
                    None,
                    timesteps=timesteps_t,
                    attention_mask=attention_mask,
                )

        if action_preds.shape[1] == 1:
            actions = action_preds[:, -1, :].cpu().numpy()
        else:
            actions = action_preds[:, -1, :].cpu().numpy()

        return actions.flatten()


def _check_shap_available():
    """Check if SHAP is available and raise informative error if not."""
    if shap is None:
        raise ImportError(
            "SHAP is not installed or not available. Please install it using:\n"
            "1. conda install -c conda-forge shap\n"
            "2. pip install shap\n"
        )


def create_ipd_feature_mappings(history_k: int = 1) -> Tuple[Dict[int, Optional[str]], Dict[Optional[str], int]]:
    """
    Create feature name mappings for IPD state features.
    
    The state structure is:
    - Base features: ["risk", "error", "delta", "infin", "contin", "r1", "r2", "r", "s", "t", "p"]
    - History features: ["my.decision1", "other.decision1", ..., "my.decision{k}", "other.decision{k}"]
    
    Args:
        history_k: Number of history decision columns to include in state (default: 1)
        
    Returns:
        Tuple of (feature_dict, feature_dict_reverse):
        - feature_dict: Maps feature index to feature name (e.g., {0: None, 1: "risk", 2: "error", ...})
        - feature_dict_reverse: Maps feature name to feature index (e.g., {None: 0, "risk": 1, "error": 2, ...})
    """
    # Base state columns (11 features)
    base_features = [
        "risk", "error", "delta", "infin", "contin", 
        "r1", "r2", "r", "s", "t", "p"
    ]
    
    # History decision columns
    history_features = []
    for k in range(1, history_k + 1):
        history_features.extend([f"my.decision{k}", f"other.decision{k}"])
    
    # Combine all features
    all_features = base_features + history_features
    
    # Create mappings following the pattern: {0: None, 1: feature1, 2: feature2, ...}
    feature_dict = {0: None}
    feature_dict_reverse = {None: 0}
    
    for idx, feature_name in enumerate(all_features, start=1):
        feature_dict[idx] = feature_name
        feature_dict_reverse[feature_name] = idx
    
    return feature_dict, feature_dict_reverse


def load_ipd_data_for_shap(
    csv_path: str,
    history_k: int = 1,
    max_trajectories: Optional[int] = None,
) -> Tuple[List[Dict], int, np.ndarray, np.ndarray]:
    """
    Load IPD data from CSV and prepare for SHAP analysis.
    
    Args:
        csv_path: Path to IPD CSV file
        history_k: Number of history decision columns to include in state
        max_trajectories: Maximum number of trajectories to load (None for all)
        
    Returns:
        Tuple of:
        - trajectories: List of trajectory dictionaries
        - max_ep_len: Maximum episode length
        - state_mean: Mean of states for normalization
        - state_std: Standard deviation of states for normalization
    """
    print(f"Loading IPD data from {csv_path}...")
    trajectories, max_ep_len = load_ipd_csv_as_trajectories(
        csv_path=csv_path,
        history_k=history_k,
    )
    
    if max_trajectories is not None and len(trajectories) > max_trajectories:
        print(f"Limiting to {max_trajectories} trajectories (out of {len(trajectories)})")
        trajectories = trajectories[:max_trajectories]
    
    print(f"Loaded {len(trajectories)} trajectories, max episode length: {max_ep_len}")
    
    # Calculate normalization statistics
    all_states = np.concatenate([traj["observations"] for traj in trajectories], axis=0)
    state_mean = all_states.mean(axis=0)
    state_std = all_states.std(axis=0) + 1e-6
    
    print(f"State dimension: {all_states.shape[1]}")
    print(f"State mean range: [{state_mean.min():.3f}, {state_mean.max():.3f}]")
    print(f"State std range: [{state_std.min():.3f}, {state_std.max():.3f}]")
    
    return trajectories, max_ep_len, state_mean, state_std


def prepare_ipd_data_for_shap(
    trajectories: List[Dict],
    K: int,
    max_ep_len: int,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    num_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare IPD trajectory data for SHAP explanation.
    
    Converts trajectories to the format expected by SHAP explainers:
    - states: (batch_size, seq_length, state_dim)
    - actions: (batch_size, seq_length, act_dim)
    - returns_to_go: (batch_size, seq_length, 1)
    - timesteps: (batch_size, seq_length)
    
    Args:
        trajectories: List of trajectory dictionaries
        K: Sequence length (max_length)
        max_ep_len: Maximum episode length
        state_mean: Mean for state normalization
        state_std: Standard deviation for state normalization
        num_samples: Number of samples to use (None for all)
        
    Returns:
        Tuple of (states, actions, returns_to_go, timesteps) as numpy arrays
    """
    def discount_cumsum(x, gamma: float = 1.0):
        """Calculate discounted cumulative sum."""
        out = np.zeros_like(x)
        out[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            out[t] = x[t] + gamma * out[t + 1]
        return out
    
    # Select samples
    if num_samples is not None and len(trajectories) > num_samples:
        selected_indices = np.random.choice(len(trajectories), num_samples, replace=False)
        selected_trajs = [trajectories[i] for i in selected_indices]
    else:
        selected_trajs = trajectories
    
    states_list = []
    actions_list = []
    returns_to_go_list = []
    timesteps_list = []
    
    state_dim = trajectories[0]["observations"].shape[1]
    act_dim = trajectories[0]["actions"].shape[1]
    
    for traj in selected_trajs:
        T = len(traj["rewards"])
        
        # If trajectory is shorter than K, just use the whole trajectory from the start
        # Otherwise, use sliding windows
        if T <= K:
            # Use the whole trajectory
            start_indices = [0]
        else:
            # Use sliding windows: take the last K timesteps for each possible start
            # Start from positions that allow at least 1 timestep
            start_indices = list(range(max(0, T - K), T))
        
        for start_idx in start_indices:
            # Extract sequence
            end_idx = min(start_idx + K, T)
            
            s = traj["observations"][start_idx:end_idx].copy()
            a = traj["actions"][start_idx:end_idx].copy()
            r = traj["rewards"][start_idx:end_idx].copy()
            
            # Normalize states
            s = (s - state_mean) / state_std
            
            # Pad to K length if needed
            if len(s) < K:
                pad_len = K - len(s)
                s_pad = np.zeros((pad_len, state_dim), dtype=np.float32)
                a_pad = np.zeros((pad_len, act_dim), dtype=np.float32)
                r_pad = np.zeros((pad_len,), dtype=np.float32)
                
                s = np.concatenate([s_pad, s], axis=0)
                a = np.concatenate([a_pad, a], axis=0)
                r = np.concatenate([r_pad, r], axis=0)
            
            # Calculate returns to go
            rtg = discount_cumsum(r, gamma=1.0)
            rtg = rtg.reshape(-1, 1)
            
            # Create timesteps
            ts = np.arange(start_idx, start_idx + K, dtype=np.int64)
            ts[ts >= max_ep_len] = max_ep_len - 1
            
            states_list.append(s)
            actions_list.append(a)
            returns_to_go_list.append(rtg)
            timesteps_list.append(ts)
    
    # Check if we have any samples
    if len(states_list) == 0:
        raise ValueError(f"No samples generated! Check that trajectories have length >= 1 and K >= 1")
    
    # Convert to numpy arrays
    states = np.array(states_list, dtype=np.float32)
    actions = np.array(actions_list, dtype=np.float32)
    returns_to_go = np.array(returns_to_go_list, dtype=np.float32)
    timesteps = np.array(timesteps_list, dtype=np.int64)
    
    print(f"Prepared {len(states_list)} samples for SHAP")
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Returns to go shape: {returns_to_go.shape}")
    print(f"Timesteps shape: {timesteps.shape}")
    
    return states, actions, returns_to_go, timesteps


def explain_ipd_decision_transformer(
    model: torch.nn.Module,
    csv_path: str,
    device: torch.device,
    history_k: int = 1,
    K: int = 40,
    background_size: int = 50,
    test_size: int = 5,
    explainer_type: str = "partition",
    max_trajectories: Optional[int] = None,
) -> Dict:
    """
    High-level function to explain IPD Decision Transformer predictions using SHAP.
    
    Args:
        model: Trained Decision Transformer model
        csv_path: Path to IPD CSV data file
        device: torch.device to run model on
        history_k: Number of history decision columns (must match training)
        K: Sequence length (must match model max_length)
        background_size: Number of background samples for SHAP
        test_size: Number of test samples to explain
        explainer_type: Type of SHAP explainer ("partition", "deep", "gradient")
        max_trajectories: Maximum number of trajectories to load (None for all)
        
    Returns:
        Dictionary containing:
        - explanation: SHAP Explanation object
        - shap_values: SHAP values
        - test_states: Test states used for explanation
        - feature_names: Feature names for visualization
        - feature_dict: Feature index to name mapping
        - feature_dict_reverse: Feature name to index mapping
    """
    _check_shap_available()
    model.eval()
    
    # Load data
    trajectories, max_ep_len, state_mean, state_std = load_ipd_data_for_shap(
        csv_path=csv_path,
        history_k=history_k,
        max_trajectories=max_trajectories,
    )
    
    # Create feature mappings
    feature_dict, feature_dict_reverse = create_ipd_feature_mappings(history_k=history_k)
    
    # Prepare data for SHAP
    states, actions, returns_to_go, timesteps = prepare_ipd_data_for_shap(
        trajectories=trajectories,
        K=K,
        max_ep_len=max_ep_len,
        state_mean=state_mean,
        state_std=state_std,
    )
    
    # Create feature names for SHAP visualization
    # Feature indices start at 1 (0 maps to None), so we map indices 1..state_dim to feature names
    feature_names = [feature_dict.get(i, f"feature_{i}") for i in range(1, states.shape[2] + 1)]
    
    # Sample background and test data
    if background_size > len(states):
        background_size = len(states)
    if test_size > len(states):
        test_size = len(states)
    
    background_indices = np.random.choice(len(states), background_size, replace=False)
    test_indices = np.random.choice(len(states), test_size, replace=False)
    
    background_states = states[background_indices]
    test_states = states[test_indices]
    test_actions = actions[test_indices]
    test_returns_to_go = returns_to_go[test_indices]
    test_timesteps = timesteps[test_indices]
    
    # Use existing SHAP explainer
    results = explain_decision_transformer(
        model=model,
        states=states,
        actions=actions,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        device=device,
        state_mean=state_mean,
        state_std=state_std,
        background_size=background_size,
        explainer_type=explainer_type,
        feature_names=feature_names,
    )
    
    # Add IPD-specific information
    results["feature_dict"] = feature_dict
    results["feature_dict_reverse"] = feature_dict_reverse
    results["ipd_trajectories"] = trajectories
    
    return results


def explain_single_ipd_prediction(
    model: torch.nn.Module,
    trajectory: Dict,
    trajectories_background: List[Dict],
    device: torch.device,
    K: int,
    max_ep_len: int,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    history_k: int = 1,
    start_idx: int = 0,
) -> Dict:
    """
    Explain a single IPD prediction from a trajectory.
    
    Args:
        model: Trained Decision Transformer model
        trajectory: Single trajectory dictionary to explain
        trajectories_background: Background trajectories for SHAP
        device: torch.device
        K: Sequence length
        max_ep_len: Maximum episode length
        state_mean: State normalization mean
        state_std: State normalization std
        history_k: Number of history decision columns
        start_idx: Starting index in trajectory to explain from
        
    Returns:
        Dictionary with SHAP values and predictions
    """
    _check_shap_available()
    model.eval()
    
    # Create feature mappings
    feature_dict, feature_dict_reverse = create_ipd_feature_mappings(history_k=history_k)
    
    def discount_cumsum(x, gamma: float = 1.0):
        """Calculate discounted cumulative sum."""
        out = np.zeros_like(x)
        out[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            out[t] = x[t] + gamma * out[t + 1]
        return out
    
    # Prepare single trajectory
    T = len(trajectory["rewards"])
    seq_len = min(K, max_ep_len) if max_ep_len is not None else K
    end_idx = min(start_idx + seq_len, T)
    
    s = trajectory["observations"][start_idx:end_idx].copy()
    a = trajectory["actions"][start_idx:end_idx].copy()
    r = trajectory["rewards"][start_idx:end_idx].copy()
    
    # Normalize states
    s = (s - state_mean) / state_std
    
    # Pad if needed
    if len(s) < seq_len:
        pad_len = seq_len - len(s)
        state_dim = s.shape[1]
        act_dim = a.shape[1]
        s_pad = np.zeros((pad_len, state_dim), dtype=np.float32)
        a_pad = np.zeros((pad_len, act_dim), dtype=np.float32)
        r_pad = np.zeros((pad_len,), dtype=np.float32)
        
        s = np.concatenate([s_pad, s], axis=0)
        a = np.concatenate([a_pad, a], axis=0)
        r = np.concatenate([r_pad, r], axis=0)
    
    rtg = discount_cumsum(r, gamma=1.0).reshape(-1, 1)
    ts = np.arange(start_idx, start_idx + seq_len, dtype=np.int64)
    if max_ep_len is not None:
        ts[ts >= max_ep_len] = max_ep_len - 1
    
    # Prepare background data
    background_states, _, _, _ = prepare_ipd_data_for_shap(
        trajectories=trajectories_background,
        K=seq_len,
        max_ep_len=max_ep_len,
        state_mean=state_mean,
        state_std=state_std,
    )
    
    # Check if background_states is empty or has wrong shape
    if background_states.size == 0:
        raise ValueError(f"Background states is empty! Check that background trajectories can generate samples with K={K}")
    
    if len(background_states.shape) < 3:
        raise ValueError(f"Background states has wrong shape: {background_states.shape}, expected (n_samples, seq_length, state_dim)")
    
    # Create wrapper
    wrapper_cls = SHAPexplainerDT if SHAPexplainerDT is not None else _IPDSHAPWrapper
    wrapper = wrapper_cls(
        model=model,
        device=device,
        state_mean=state_mean,
        state_std=state_std,
        baseline_actions=a[np.newaxis, :, :],
        baseline_returns_to_go=rtg[np.newaxis, :, :],
        max_ep_len=max_ep_len,
    )
    # Ensure wrapper uses the same sequence length used here
    wrapper.max_length = seq_len
    
    # For SHAP, we explain features at the last timestep (where the prediction is made)
    # We need to create a wrapper function that takes only the last timestep features
    # and reconstructs the full sequence for the model
    
    def predict_last_timestep_features(features_array):
        """
        Wrapper function for SHAP that takes last timestep features and returns predictions.
        
        Args:
            features_array: numpy array of shape (n_samples, state_dim) - features at last timestep
                           OR shape (state_dim,) - single sample
        
        Returns:
            numpy array of shape (n_samples,) with predictions
        """
        # Handle both single sample and batch
        if features_array.ndim == 1:
            features_array = features_array[np.newaxis, :]  # (1, state_dim)
        
        n_samples = features_array.shape[0]
        state_dim = features_array.shape[1]
        seq_length = s.shape[0]  # Use sequence length from the trajectory we're explaining
        
        # Reconstruct full sequences: use s for all timesteps except the last
        # Replace last timestep with the features from SHAP
        full_sequences = np.tile(s, (n_samples, 1, 1))  # (n_samples, seq_length, state_dim)
        full_sequences[:, -1, :] = features_array  # Replace last timestep
        
        # Flatten for the wrapper's predict function
        flattened = full_sequences.reshape(n_samples, seq_length * state_dim)
        
        # Call wrapper predict
        predictions = wrapper.predict(flattened)
        
        return predictions
    
    # Extract last timestep states: shape (1, state_dim)
    states_last_timestep = np.asarray(s[-1, :], dtype=np.float32).reshape(1, -1)
    
    # Extract last timestep from background: shape (n_samples, state_dim)
    background_last_timestep = np.asarray(background_states[:, -1, :], dtype=np.float32)
    background_last_timestep = np.atleast_2d(background_last_timestep)
    
    # Verify shapes
    if states_last_timestep.shape[1] != background_last_timestep.shape[1]:
        raise ValueError(
            f"State dimension mismatch! "
            f"Test state: {states_last_timestep.shape}, "
            f"Background: {background_last_timestep.shape}"
        )
    
    # Use PartitionExplainer - it expects background of shape (n_samples, n_features)
    # and test data of shape (n_features,)
    _check_shap_available()
    explainer = shap.PartitionExplainer(predict_last_timestep_features, background_last_timestep)
    shap_values_flat = explainer(states_last_timestep)
    
    # shap_values_flat.values should have shape (state_dim,)
    # We'll create a full SHAP values array with zeros for earlier timesteps
    shap_values = np.zeros((s.shape[0], s.shape[1]), dtype=np.float32)
    if hasattr(shap_values_flat, 'values'):
        shap_values[-1, :] = shap_values_flat.values[0]
    else:
        shap_values[-1, :] = np.asarray(shap_values_flat)[0]
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        states_batch = s[np.newaxis, :, :]  # (1, seq_length, state_dim)
        actions_batch = a[np.newaxis, :, :]  # (1, seq_length, act_dim)
        returns_to_go_batch = rtg[np.newaxis, :, :]  # (1, seq_length, 1)
        timesteps_batch = ts[np.newaxis, :]  # (1, seq_length)
        
        states_t = torch.as_tensor(states_batch, dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions_batch, dtype=torch.float32, device=device)
        returns_to_go_t = torch.as_tensor(returns_to_go_batch, dtype=torch.float32, device=device)
        timesteps_t = torch.as_tensor(timesteps_batch, dtype=torch.long, device=device)
        
        forward_sig = inspect.signature(model.forward)
        forward_params = list(forward_sig.parameters.keys())
        
        if 'returns_to_go' in forward_params:
            # Align returns_to_go length with sequence length
            seq_len_actual = states_t.shape[1]
            if returns_to_go_t.shape[1] == seq_len_actual + 1:
                returns_to_go_aligned = returns_to_go_t[:, :-1]
            elif returns_to_go_t.shape[1] == seq_len_actual:
                returns_to_go_aligned = returns_to_go_t
            else:
                if returns_to_go_t.shape[1] > seq_len_actual:
                    returns_to_go_aligned = returns_to_go_t[:, :seq_len_actual]
                else:
                    pad_len = seq_len_actual - returns_to_go_t.shape[1]
                    pad = torch.zeros(
                        (returns_to_go_t.shape[0], pad_len, 1),
                        dtype=returns_to_go_t.dtype,
                        device=returns_to_go_t.device,
                    )
                    returns_to_go_aligned = torch.cat([returns_to_go_t, pad], dim=1)

            _, action_pred, _ = model.forward(
                states_t, actions_t, None,
                returns_to_go_aligned, timesteps_t,
                attention_mask=torch.ones((1, states_t.shape[1]), device=device)
            )
        else:
            _, action_pred, _ = model.forward(
                states_t, actions_t, None,
                timesteps=timesteps_t,
                attention_mask=torch.ones((1, states_t.shape[1]), device=device)
            )
        
        predicted_action = action_pred[0, -1].cpu().numpy()
    
    # Get predicted action label (defect=0 or coop=1)
    pred_prob = torch.sigmoid(torch.tensor(predicted_action[0]))
    pred_label = 1 if pred_prob >= 0.5 else 0
    
    results = {
        "shap_values": shap_values,
        "predicted_action": predicted_action,
        "predicted_label": pred_label,
        "predicted_prob": float(pred_prob),
        "predicted_label_name": "coop" if pred_label == 1 else "defect",
        "states": s,
        "feature_dict": feature_dict,
        "feature_dict_reverse": feature_dict_reverse,
        "trajectory": trajectory,
        "start_idx": start_idx,
        "explanation": shap_values_flat,
    }
    
    return results


def explain_random_ipd_trajectories(
    model: torch.nn.Module,
    trajectories: List[Dict],
    device: torch.device,
    K: int,
    max_ep_len: int,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    history_k: int = 1,
    num_trajectories: int = 10,
    background_size: int = 50,
) -> List[Dict]:
    """
    Randomly select and explain predictions from multiple IPD trajectories.
    
    Args:
        model: Trained Decision Transformer model
        trajectories: List of all trajectory dictionaries
        device: torch.device
        K: Sequence length
        max_ep_len: Maximum episode length
        state_mean: State normalization mean
        state_std: State normalization std
        history_k: Number of history decision columns
        num_trajectories: Number of random trajectories to analyze (default: 10)
        background_size: Number of background trajectories for SHAP
        
    Returns:
        List of dictionaries, each containing SHAP explanation for one trajectory
    """
    _check_shap_available()
    
    # Randomly select trajectories
    if num_trajectories > len(trajectories):
        num_trajectories = len(trajectories)
    
    selected_indices = np.random.choice(len(trajectories), num_trajectories, replace=False)
    selected_trajs = [trajectories[i] for i in selected_indices]
    
    # Use remaining trajectories as background
    background_indices = [i for i in range(len(trajectories)) if i not in selected_indices]
    if len(background_indices) > background_size:
        background_indices = np.random.choice(background_indices, background_size, replace=False)
    background_trajs = [trajectories[i] for i in background_indices]
    
    print(f"Analyzing {num_trajectories} random trajectories...")
    print(f"Using {len(background_trajs)} trajectories as background")
    
    results_list = []
    
    for idx, traj in enumerate(selected_trajs):
        print(f"Processing trajectory {idx + 1}/{num_trajectories}...")
        
        # Explain from the last timestep (most recent prediction)
        T = len(traj["rewards"])
        start_idx = max(0, T - K)
        
        result = explain_single_ipd_prediction(
            model=model,
            trajectory=traj,
            trajectories_background=background_trajs,
            device=device,
            K=K,
            max_ep_len=max_ep_len,
            state_mean=state_mean,
            state_std=state_std,
            history_k=history_k,
            start_idx=start_idx,
        )
        
        results_list.append(result)
    
    return results_list


def explain_ipd_trajectories_in_batches(
    model: torch.nn.Module,
    trajectories: List[Dict],
    device: torch.device,
    K: int,
    max_ep_len: int,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    history_k: int = 1,
    total_trajectories: int = 1000,
    batch_trajectories: int = 100,
    background_size: int = 50,
    seed: Optional[int] = None,
) -> List[List[Dict]]:
    """
    Analyze trajectories in multiple batches while keeping normalization fixed.

    This will:
    1) Randomly select `total_trajectories` from the dataset
    2) Split them into batches of size `batch_trajectories`
    3) Run SHAP analysis for each batch

    Args:
        model: Trained Decision Transformer model
        trajectories: List of all trajectory dictionaries
        device: torch.device
        K: Sequence length
        max_ep_len: Maximum episode length
        state_mean: State normalization mean (fixed across all batches)
        state_std: State normalization std (fixed across all batches)
        history_k: Number of history decision columns
        total_trajectories: Total number of trajectories to analyze
        batch_trajectories: Number of trajectories per analysis batch
        background_size: Number of background trajectories for SHAP (per batch)
        seed: Optional random seed for reproducibility

    Returns:
        List of batches, each batch is a list of result dicts
    """
    _check_shap_available()

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    total_trajectories = min(total_trajectories, len(trajectories))
    if total_trajectories <= 0:
        raise ValueError("total_trajectories must be > 0")

    if batch_trajectories <= 0:
        raise ValueError("batch_trajectories must be > 0")

    # Sample a pool of trajectories without replacement
    pool_indices = rng.choice(len(trajectories), size=total_trajectories, replace=False)
    pool_indices = pool_indices.tolist()

    num_batches = int(np.ceil(total_trajectories / batch_trajectories))
    batch_results: List[List[Dict]] = []

    print(f"Analyzing {total_trajectories} trajectories in {num_batches} batches "
          f"of {batch_trajectories}...")

    for batch_idx in range(num_batches):
        start = batch_idx * batch_trajectories
        end = min(start + batch_trajectories, total_trajectories)
        batch_indices = pool_indices[start:end]
        batch_trajs = [trajectories[i] for i in batch_indices]

        # Background from remaining trajectories not in this batch
        remaining_indices = [i for i in range(len(trajectories)) if i not in batch_indices]
        if len(remaining_indices) > background_size:
            background_indices = rng.choice(remaining_indices, size=background_size, replace=False)
        else:
            background_indices = remaining_indices
        background_trajs = [trajectories[i] for i in background_indices]

        print(f"\nBatch {batch_idx + 1}/{num_batches}: "
              f"{len(batch_trajs)} trajectories, {len(background_trajs)} background")

        batch_list: List[Dict] = []
        for i, traj in enumerate(batch_trajs):
            T = len(traj["rewards"])
            start_idx = max(0, T - K)
            result = explain_single_ipd_prediction(
                model=model,
                trajectory=traj,
                trajectories_background=background_trajs,
                device=device,
                K=K,
                max_ep_len=max_ep_len,
                state_mean=state_mean,
                state_std=state_std,
                history_k=history_k,
                start_idx=start_idx,
            )
            batch_list.append(result)

        batch_results.append(batch_list)

    return batch_results


# ============================================================================
# Visualization Functions
# ============================================================================

def bar_chart_explanation(tokenized_text, values, class_to_explain, pred):
    """
    Create a bar chart showing SHAP values for each feature.
    
    Args:
        tokenized_text: List of feature names (or tokenized text)
        values: List of SHAP values corresponding to each feature
        class_to_explain: Name of the class being explained (e.g., "coop" or "defect")
        pred: Prediction probability (0-1 scale)
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization")
    
    values = np.array(values)
    plt.figure(figsize=(12, 6))
    
    colors = ["green" if x > 0 else "red" for x in values]
    plt.bar([*range(len(values))], values, color=colors)
    plt.xticks(np.arange(len(tokenized_text)), tokenized_text, fontsize=15)
    plt.yticks(fontsize=15)
    plt.axhline(y=0, color='black', linestyle='dashed')
    title = f"Predicted class: {class_to_explain} ({pred:.2f} %)"
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.show()


def text_box_explanation(raw, values):
    """
    Create a text box visualization with colored backgrounds based on SHAP values.
    
    Args:
        raw: List of raw feature names/text
        values: List of SHAP values corresponding to each feature
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization")
    
    values = np.array(values)
    fixed_y = 0.5
    fig, ax = plt.subplots(figsize=(12, 6))
    Yy = 5
    plt.xlim((0, Yy))
    plt.ylim((0.4, 0.6))
    threshold = sum(abs(values)) * 0.01
    h = [x if abs(x) > threshold else 0 for x in values]
    h /= np.sum(np.abs(h))
    show_box = [["green", "mediumaquamarine"][int(x * 10 < 1)] if abs(x) > threshold and x > 0 else
                ["red", "tomato"][int(x * 10 > -1)] if abs(x) > threshold else "white" for x in h]
    text_color = ["white" if abs(x) > threshold and x > 0 else "black" for x in values]
    coord = []
    for i, word in enumerate(raw):
        x = 0 if not coord else coord[-1][1]
        t = plt.text(x=x, y=fixed_y, s=word, ha="center", va="center", size=20, rotation=0., color=text_color[i],
                     bbox=dict(boxstyle="square", ec="white", fc=show_box[i], ))
        tt = t.get_window_extent(renderer=fig.canvas.get_renderer())
        transf = ax.transData.inverted()
        d = tt.transformed(transf)
        f = (d.x0, d.x1, d.y0, d.y1)
        diff_x = d.x1 - d.x0 + 0.01
        
        if not coord:
            t.set_position((diff_x / 2, fixed_y))
        elif x + diff_x < Yy:
            t.set_position((x + diff_x / 2, fixed_y))
        else:
            fixed_y -= 0.1
            t.set_position((diff_x / 2, fixed_y))
        coord.append((t.get_position(), x + diff_x))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def determine_graph_width(max_word, max_length):
    """Determine graph width based on maximum word length and sequence length."""
    return max_word * max_length * 0.15


def joint_visualization(tokenized_text, values, class_to_explain, pred, i, save_path=None):
    """
    Create a joint bar chart visualization with colored bars based on SHAP values.
    
    Args:
        tokenized_text: List of feature names
        values: List of SHAP values corresponding to each feature
        class_to_explain: Name of the class being explained (e.g., "coop" or "defect")
        pred: Prediction probability (0-1 scale)
        i: Index/identifier for this visualization
        save_path: Optional path to save the figure (if None, shows the plot)
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization")
    
    if sns is None:
        raise ImportError("seaborn is required for joint_visualization")
    
    font_size = 14
    font_properties = {'family': 'serif', 'serif': ['Computer Modern Roman'],
                       'weight': 'normal', 'size': font_size}
    
    font_manager.FontProperties(family='Computer Modern Roman', style='normal',
                                size=font_size, weight='normal', stretch='normal')
    rc('font', **font_properties)
    
    sns.set_style('whitegrid')
    plt.rcParams["figure.figsize"] = (determine_graph_width(max_word=len(max(tokenized_text, key=len)), max_length=len(tokenized_text)), 6)
    fig, ax = plt.subplots(1, 1)
    values = np.array(values)
    
    perc_pos, perc_neg = 0, 0
    for xx in [[x for x in values if x > 0], [x for x in values if x <= 0]]:
        try:
            if all([x > 0 for x in xx]):
                perc_pos = np.percentile(xx, 50)
            elif all([x <= 0 for x in xx]):
                perc_neg = np.percentile(xx, 50)
        except IndexError:
            pass
    
    colors = [["mediumaquamarine", "green"][int(x > perc_pos)] if x > 0 else
              ["red", "tomato"][int(x > perc_neg)] for x in values]
    
    plt.bar([*range(len(values))], values, color=colors, edgecolor="black", alpha=0.6)
    ax.set_xticks([*range(len(values))])
    ax.set_xticklabels(tokenized_text)
    colors_ticks = colors
    for ticklabel, tickcolor in zip(ax.get_xticklabels(), colors_ticks):
        bbox = dict(boxstyle="round", ec="black", fc=tickcolor, alpha=0.2)
        plt.setp(ticklabel, bbox=bbox)
    plt.axhline(y=0, color='black', linestyle='dashed')
    pred *= 100
    title = f"Predicted class: {class_to_explain} ({pred:.2f} %)"
    fig.suptitle(title)
    plt.ylabel("Impact on model output")
    pname = "".join(tokenized_text[0:3])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"DONE {save_path}")
        plt.close()
    else:
        plt.show()


def visualize_ipd_shap_explanation(
    result: Dict,
    visualization_type: str = "joint",
    save_path: Optional[str] = None,
    feature_idx: int = -1,  # Which timestep to visualize (-1 for last)
):
    """
    Visualize SHAP explanation for a single IPD prediction.
    
    Args:
        result: Result dictionary from explain_single_ipd_prediction
        visualization_type: Type of visualization ("joint", "bar", "text_box")
        save_path: Optional path to save the figure
        feature_idx: Which timestep's features to visualize (-1 for last timestep)
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization")
    
    shap_values = result["shap_values"]  # (seq_length, state_dim)
    feature_dict = result["feature_dict"]
    pred_label_name = result.get("predicted_label_name", "unknown")
    pred_prob = result.get("predicted_prob", 0.0)
    
    # Select timestep to visualize
    if feature_idx == -1:
        feature_idx = shap_values.shape[0] - 1
    feature_idx = min(feature_idx, shap_values.shape[0] - 1)
    
    # Get SHAP values and feature names for this timestep
    values = shap_values[feature_idx, :].tolist()
    feature_names = [feature_dict.get(i + 1, f"feature_{i + 1}") for i in range(len(values))]
    
    # Filter out None values if any
    feature_names = [name if name is not None else f"feature_{i + 1}" 
                     for i, name in enumerate(feature_names)]
    
    if visualization_type == "joint":
        save_path_actual = save_path if save_path else f"figures/our_vis_{feature_idx}.png"
        joint_visualization(feature_names, values, pred_label_name, pred_prob, feature_idx, save_path_actual)
    elif visualization_type == "bar":
        bar_chart_explanation(feature_names, values, pred_label_name, pred_prob * 100)
    elif visualization_type == "text_box":
        text_box_explanation(feature_names, values)
    else:
        raise ValueError(f"Unknown visualization type: {visualization_type}")


if __name__ == "__main__":
    # Example usage
    print("IPD SHAP Explainer")
    print("=" * 60)
    print("\nThis module provides utilities for explaining IPD Decision Transformer models.")
    print("\nExample usage:")
    print("""
    import torch
    from decision_transformer.explainability.shap_explainer_ipd import (
        explain_ipd_decision_transformer,
        create_ipd_feature_mappings,
        load_ipd_data_for_shap,
    )
    
    # Create feature mappings
    feature_dict, feature_dict_reverse = create_ipd_feature_mappings(history_k=1)
    print(f"Feature mappings: {feature_dict}")
    
    # Load model (example)
    # model = torch.load("path/to/model.pt")
    
    # Explain model
    # results = explain_ipd_decision_transformer(
    #     model=model,
    #     csv_path="data/all_data.csv",
    #     device=torch.device("cuda"),
    #     history_k=1,
    #     K=40,
    # )
    """)
    
    # Test feature mappings
    print("\nTesting feature mappings...")
    feature_dict, feature_dict_reverse = create_ipd_feature_mappings(history_k=3)
    print(f"Number of features: {len(feature_dict) - 1}")  # -1 because 0 maps to None
    print(f"First 5 features: {[feature_dict[i] for i in range(1, 6)]}")
    print(f"Feature 'risk' maps to index: {feature_dict_reverse.get('risk')}")
