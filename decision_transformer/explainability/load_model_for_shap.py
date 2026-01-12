"""
Example script showing how to load a saved Decision Transformer model for SHAP explanation.

This demonstrates:
1. Loading a saved model checkpoint
2. Preparing data for SHAP
3. Running SHAP explanations
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.gpt2_bc import GPT2BCModel
from decision_transformer.explainability.shap_explainer import (
    explain_decision_transformer,
    explain_single_prediction,
)


def load_decision_transformer_model(
    model_path: str,
    state_dim: int,
    act_dim: int,
    hidden_size: int = 128,
    max_length: int = 20,
    device: torch.device = None,
) -> torch.nn.Module:
    """
    Load a saved Decision Transformer model from checkpoint.
    
    Args:
        model_path: Path to saved model checkpoint (.pt or .pth file)
        state_dim: State dimension
        act_dim: Action dimension
        hidden_size: Hidden size (must match training config)
        max_length: Max sequence length (must match training config)
        device: Device to load model on (default: cuda if available, else cpu)
        
    Returns:
        Loaded model in eval mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model with same architecture as training
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=hidden_size,
        max_length=max_length,
    )
    
    # Load saved weights
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        # If checkpoint is a dict, it might have 'model_state_dict' or just be the state_dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume the whole dict is the state_dict
            model.load_state_dict(checkpoint)
    else:
        # Direct state_dict
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded successfully on {device}")
    return model


def load_gpt2_bc_model(
    model_path: str,
    state_dim: int,
    act_dim: int,
    hidden_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    max_length: int = 20,
    device: torch.device = None,
) -> torch.nn.Module:
    """
    Load a saved GPT2BCModel from checkpoint.
    
    Args:
        model_path: Path to saved model checkpoint
        state_dim: State dimension
        act_dim: Action dimension
        hidden_size: Hidden size
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        max_length: Max sequence length
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GPT2BCModel(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=hidden_size,
        n_layer=n_layer,
        n_head=n_head,
        max_length=max_length,
    )
    
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model


def save_model_example(model: torch.nn.Module, save_path: str, save_full_checkpoint: bool = False):
    """
    Example of how to save a model for later loading.
    
    Args:
        model: Trained model
        save_path: Path to save the model
        save_full_checkpoint: If True, save full checkpoint with metadata
    """
    if save_full_checkpoint:
        # Save full checkpoint (useful if you want to save optimizer state, epoch, etc.)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'state_dim': model.state_dim if hasattr(model, 'state_dim') else None,
            'act_dim': model.act_dim if hasattr(model, 'act_dim') else None,
            # Add other metadata as needed
        }
        torch.save(checkpoint, save_path)
    else:
        # Save just the state dict (simpler, smaller file)
        torch.save(model.state_dict(), save_path)
    
    print(f"Model saved to {save_path}")


def example_using_saved_model():
    """
    Example: Load a saved model and use SHAP to explain it.
    """
    print("=" * 60)
    print("Example: Using SHAP with a Saved Model")
    print("=" * 60)
    
    # Configuration (should match your training config)
    model_path = "models/decision_transformer_model.pt"  # Update with your path
    state_dim = 17  # Update with your state dimension
    act_dim = 1     # Update with your action dimension
    hidden_size = 128
    max_length = 20
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    try:
        model = load_decision_transformer_model(
            model_path=model_path,
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_size=hidden_size,
            max_length=max_length,
            device=device,
        )
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        print("Please update model_path with the correct path to your saved model.")
        print("\nTo save a model during training, use:")
        print("  torch.save(model.state_dict(), 'models/my_model.pt')")
        return
    
    # Load your data (replace with actual data loading)
    # You need: states, actions, returns_to_go, timesteps
    # These should be numpy arrays from your dataset
    print("\nLoading data...")
    # Example: Load from your data source
    # states = np.load("data/states.npy")
    # actions = np.load("data/actions.npy")
    # returns_to_go = np.load("data/returns_to_go.npy")
    # timesteps = np.load("data/timesteps.npy")
    
    # For demonstration, create dummy data
    print("Using dummy data for demonstration...")
    batch_size = 10
    states = np.random.randn(batch_size, max_length, state_dim).astype(np.float32)
    actions = np.random.randn(batch_size, max_length, act_dim).astype(np.float32)
    returns_to_go = np.random.randn(batch_size, max_length, 1).astype(np.float32)
    timesteps = np.tile(np.arange(max_length, dtype=np.int64), (batch_size, 1))
    
    # Normalization parameters (should match your training normalization)
    state_mean = np.zeros(state_dim)  # Update with actual mean from training
    state_std = np.ones(state_dim)     # Update with actual std from training
    
    print("\nRunning SHAP explanation...")
    
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
    )
    
    print(f"\nSHAP explanation completed!")
    print(f"SHAP values shape: {results['shap_values'].shape}")
    print(f"Test states shape: {results['test_states'].shape}")


def example_using_in_memory_model():
    """
    Example: Use SHAP with a model that's already in memory (e.g., just finished training).
    """
    print("\n" + "=" * 60)
    print("Example: Using SHAP with In-Memory Model")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Assume you just finished training and have the model in memory
    print("Creating/loading model (in your case, this would be your trained model)...")
    model = DecisionTransformer(
        state_dim=17,
        act_dim=1,
        hidden_size=128,
        max_length=20,
    ).to(device)
    model.eval()
    
    # You can use SHAP directly without saving/loading
    print("\nUsing SHAP with in-memory model (no save/load needed)...")
    
    # Prepare data
    states = np.random.randn(5, 20, 17).astype(np.float32)
    actions = np.random.randn(5, 20, 1).astype(np.float32)
    returns_to_go = np.random.randn(5, 20, 1).astype(np.float32)
    timesteps = np.tile(np.arange(20, dtype=np.int64), (5, 1))
    
    results = explain_decision_transformer(
        model=model,
        states=states,
        actions=actions,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        device=device,
        state_mean=np.zeros(17),
        state_std=np.ones(17),
        background_size=3,
    )
    
    print("SHAP explanation completed with in-memory model!")


if __name__ == "__main__":
    print("SHAP with Saved vs In-Memory Models")
    print("=" * 60)
    
    # Example 1: Using a saved model
    example_using_saved_model()
    
    # Example 2: Using an in-memory model
    example_using_in_memory_model()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print("1. You DON'T need to save the model if it's already in memory")
    print("2. You DO need to save the model if you want to explain it later")
    print("3. When loading, make sure architecture matches training config")
    print("4. Always set model.eval() before using SHAP")
    print("=" * 60)

