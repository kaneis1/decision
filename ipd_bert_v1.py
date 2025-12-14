## Python Imports

import argparse
import os
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import init
import seaborn as sns
import math
import wandb
from typing import Dict, List

sns.set()
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression

MSE = mean_squared_error

import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)

## Plotting Config

SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 30

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

## Data Loading Functions (from experiment_ipd_bert.py)

def load_ipd_csv_as_trajectories(csv_path: str, history_k: int = 1):
    """
    Load IPD trajectories from CSV.
    
    Args:
        csv_path: Path to CSV file
        history_k: Number of history decision columns to include (default: 1)
                   When history_k=3, includes: my.decision1, other.decision1, 
                   my.decision2, other.decision2, my.decision3, other.decision3
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]

    period_col = "period"
    action_col = "my.decision"

    # Build state columns: base + decision history + payoff history
    state_cols = [
        "risk", "error", "delta", "infin", "contin", "r1", "r2", "r", "s", "t", "p"
    ]
    # Add decision history columns
    for k in range(1, history_k + 1):
        state_cols.extend([f"my.decision{k}", f"other.decision{k}"])
    # Add payoff history columns
    for k in range(1, history_k + 1):
        state_cols.extend([f"my.payoff{k}", f"other.payoff{k}"])

    groups = []
    start_idx = 0

    for i in range(0, len(df["data_id"]) - 1):  
        if df["period"].iloc[i] != df["period"].iloc[i + 1] - 1:
            groups.append(df.iloc[start_idx:i+1].copy())
            start_idx = i+1
    groups.append(df.iloc[start_idx:].copy())
    
    trajectories, max_ep_len = [], 1
    action_map = {"coop": 1, "defect": 0}

    for ep in groups:   
        ep[action_col] = ep[action_col].map(action_map)

        # Fill NaN values for all history decision columns (set to 2 for missing values)
        for k in range(1, history_k + 1):
            my_col = f"my.decision{k}"
            other_col = f"other.decision{k}"
            if my_col in ep.columns:
                ep[my_col] = ep[my_col].fillna(2)
            if other_col in ep.columns:
                ep[other_col] = ep[other_col].fillna(2)
        # Fill NaN values for payoff history columns (set to 0 for missing values)
        for k in range(1, history_k + 1):
            my_payoff_col = f"my.payoff{k}"
            other_payoff_col = f"other.payoff{k}"
            if my_payoff_col in ep.columns:
                ep[my_payoff_col] = ep[my_payoff_col].fillna(0.0)
            if other_payoff_col in ep.columns:
                ep[other_payoff_col] = ep[other_payoff_col].fillna(0.0)
        state_vals = (
            ep[state_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .astype(float)
            .values
        )
        s = state_vals.astype(np.float32)
        a = ep[action_col].values.astype(np.float32).reshape(-1, 1)
        r = ep["my.payoff1"].fillna(0.0).values.astype(np.float32)

        T = len(ep)
        terminals = np.zeros(T, dtype=np.int64)
        terminals[-1] = 1

        trajectories.append(dict(observations=s, actions=a, rewards=r, terminals=terminals, lens=int(T)))
        max_ep_len = max(max_ep_len, T)

    return trajectories, max_ep_len


def discount_cumsum(x, gamma: float):
    """Compute discounted cumulative sum (for RTG computation)."""
    out = np.zeros_like(x)
    out[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        out[t] = x[t] + gamma * out[t + 1]
    return out


def extract_trajectories_for_dt(trajectories: List[Dict], history_k: int = 1):
    """
    Extract trajectories for Decision Transformer (with RTG).
    
    Args:
        trajectories: List of trajectory dicts from load_ipd_csv_as_trajectories
        history_k: Number of history steps to include
    
    Returns:
        dt_trajectories: List of dict with 'states', 'actions', 'rewards', 'returns' keys
        max_ep_len: int
    """
    dt_trajectories = []
    max_ep_len = 1
    
    for traj in trajectories:
        observations = traj["observations"]  # [T, state_dim]
        actions = traj["actions"]  # [T, 1]
        rewards = traj["rewards"]  # [T]
        T = observations.shape[0]
        
        # Compute returns-to-go (RTG) from rewards
        returns = discount_cumsum(rewards, gamma=1.0)  # [T] - RTG at each timestep
        
        # States are already in the correct format
        states = observations.astype(np.float32)  # [T, state_dim]
        actions_flat = actions.reshape(-1).astype(np.float32)  # [T]
        rewards_flat = rewards.astype(np.float32)  # [T]
        returns_flat = returns.astype(np.float32)  # [T]
        
        max_ep_len = max(max_ep_len, T)
        dt_trajectories.append({
            'states': states, 
            'actions': actions_flat,
            'rewards': rewards_flat,
            'returns': returns_flat,
            'lens': T
        })
    
    return dt_trajectories, max_ep_len

## BERT-like Transformer Components

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Store as (max_len, 1, d_model) for broadcasting
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        seq_len = x.size(0)
        # Get positional encoding for this sequence length and broadcast to batch
        pe = self.pe[:seq_len, :, :]  # (seq_len, 1, d_model)
        return x + pe  # Broadcasting: (seq_len, batch_size, d_model) + (seq_len, 1, d_model)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(attention_output)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

## Models

class lstmModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layer_num):
        super().__init__()
        self.lstmLayer = nn.LSTM(in_dim, hidden_dim, layer_num)
        self.relu = nn.ReLU()
        self.fcLayer = nn.Linear(hidden_dim, out_dim)
        self.weightInit = np.sqrt(1.0 / hidden_dim)

    def forward(self, x):
        out, _ = self.lstmLayer(x)
        out = self.relu(out)
        out = self.fcLayer(out)
        out = nn.Softmax(dim=-1)(out)
        return out


class bertModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_layers, num_heads=2, dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.state_dim = state_dim
        
        # Input embedding for states
        self.input_embedding = nn.Linear(state_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=max_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer: binary classification (1 logit)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, state_dim)
        batch_size, seq_len, _ = x.shape
        
        # Input embedding
        x = self.input_embedding(x)  # (batch_size, seq_len, hidden_dim)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Output layer: returns logits for binary classification
        x = self.output_layer(x)  # (batch_size, seq_len, 1)
        
        return x

## Evaluation Metrics (from experiment_ipd_bert.py)

def evaluate_bert_ipd_metrics(
    model,
    trajectories: List[Dict],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    state_dim: int,
    device: torch.device,
    K: int,
    max_ep_len: int,
    print_report: bool = True,
) -> Dict[str, float]:
    """Evaluate BERT model with same metrics as experiment_ipd_bert.py."""
    model.eval()
    
    def _sigmoid_safe(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.sigmoid(x), 1e-6, 1 - 1e-6)
    
    def _binary_ll(y_true: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return (y_true * torch.log(p) + (1 - y_true) * torch.log(1 - p)).sum()
    
    y_t1, p_t1 = [], []
    y_tgt, p_tgt = [], []
    per_time_truth, per_time_pred = {}, {}
    
    with torch.no_grad():
        for traj_idx, traj in enumerate(trajectories):
            states = traj['states']  # [T, state_dim]
            actions_true = traj['actions']  # [T]
            T = states.shape[0]
            
            # Normalize states
            states_norm = (states - state_mean) / state_std
            
            # Process trajectory with sliding window of size K
            for start_idx in range(T):
                seq_len = min(K, T - start_idx)
                if seq_len <= 0:
                    continue
                
                # Extract sequence window
                s_window = states_norm[start_idx:start_idx + seq_len]  # [seq_len, state_dim]
                a_window = actions_true[start_idx:start_idx + seq_len]  # [seq_len]
                
                # Pad to K if needed
                if seq_len < K:
                    s_pad = np.zeros((K - seq_len, state_dim), dtype=np.float32)
                    s_padded = np.concatenate([s_pad, s_window], axis=0).astype(np.float32)  # [K, state_dim]
                    mask = np.concatenate([np.zeros(K - seq_len), np.ones(seq_len)]).astype(np.float32)  # [K]
                else:
                    s_padded = s_window.astype(np.float32)
                    mask = np.ones(seq_len, dtype=np.float32)
                
                # Convert to tensors
                states_tensor = torch.tensor(s_padded, dtype=torch.float32, device=device).unsqueeze(0)  # [1, K, state_dim]
                attention_mask = torch.tensor(mask, dtype=torch.long, device=device).unsqueeze(0)  # [1, K]
                
                # Get predictions
                logits = model(states_tensor, mask=attention_mask)  # [1, K, 1]
                
                # Get prediction for the last timestep (current action)
                if seq_len > 0:
                    logit_last = logits[0, -1, 0]  # Last timestep prediction
                    prob_last = torch.sigmoid(logit_last).cpu().item()
                    
                    # Get true action for this timestep
                    true_action_idx = start_idx + seq_len - 1
                    if true_action_idx < T:
                        y_true = float(actions_true[true_action_idx])
                        
                        # Store predictions
                        t_abs = true_action_idx + 1  # 1-based absolute time
                        if t_abs == 1:
                            y_t1.append(y_true)
                            p_t1.append(prob_last)
                        else:
                            y_tgt.append(y_true)
                            p_tgt.append(prob_last)
                        
                        per_time_truth.setdefault(t_abs, []).append(y_true)
                        per_time_pred.setdefault(t_abs, []).append(prob_last)
    
    def _acc_ll(y_list, p_list):
        if not y_list:
            return 0.0, 0.0
        y = torch.tensor(y_list, dtype=torch.float32)
        p = torch.tensor(p_list, dtype=torch.float32)
        acc = ((p >= 0.5).float() == y).float().mean().item()
        ll = _binary_ll(y, p).item()
        return acc, ll
    
    acc_t1, ll_t1 = _acc_ll(y_t1, p_t1)
    acc_tgt, ll_tgt = _acc_ll(y_tgt, p_tgt)
    
    def _agg(d_truth: Dict[int, List[float]], d_pred: Dict[int, List[float]]):
        keys = sorted(set(d_truth.keys()) & set(d_pred.keys()))
        if not keys:
            return 0.0, 0.0
        truth = np.array([np.mean(d_truth[k]) for k in keys], dtype=np.float64)
        pred = np.array([np.mean(d_pred[k]) for k in keys], dtype=np.float64)
        corr = 0.0 if (truth.std() < 1e-12 or pred.std() < 1e-12) else float(np.corrcoef(truth, pred)[0, 1])
        rmse = float(np.sqrt(np.mean((truth - pred) ** 2)))
        return corr, rmse
    
    cor_time, rmse_time = _agg(per_time_truth, per_time_pred)
    cor_avg, rmse_avg = 0.0, 0.0  # Not applicable for this model
    
    report = {
        "Acc.t=1": acc_t1, "Acc.t>1": acc_tgt,
        "LL.t=1": ll_t1, "LL.t>1": ll_tgt,
        "Cor-Time": cor_time, "RMSE-Time": rmse_time,
        "Cor-Avg": cor_avg, "RMSE-Avg": rmse_avg,
    }
    
    if print_report:
        metrics_str = ", ".join([
            f"Acc.t=1 {acc_t1:.3f}", f"Acc.t>1 {acc_tgt:.3f}",
            f"LL.t=1 {ll_t1:.0f}", f"LL.t>1 {ll_tgt:.0f}",
            f"Cor-Time {cor_time:.3f}", f"RMSE-Time {rmse_time:.3f}",
            f"Cor-Avg {cor_avg:.3f}", f"RMSE-Avg {rmse_avg:.3f}",
        ])
        print("BERT IPD Eval →", metrics_str)
    
    return report


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    # Load CSV -> trajectories (same as experiment_ipd_bert.py)
    full_trajectories, max_ep_len = load_ipd_csv_as_trajectories(
        csv_path=args.csv_path,
        history_k=args.history_k,
    )
    
    # Extract trajectories for Decision Transformer (with RTG)
    trajectories, max_ep_len = extract_trajectories_for_dt(
        full_trajectories,
        history_k=args.history_k,
    )
    
    states = np.concatenate([tr['states'] for tr in trajectories], axis=0)
    state_dim = states.shape[1]
    traj_lens = np.array([tr["lens"] for tr in trajectories], dtype=np.int32)
    num_timesteps = int(traj_lens.sum())
    total_trajs = len(trajectories)
    
    print("=" * 50)
    print(f"IPD-CSV dataset (BERT): {total_trajs} trajectories, {num_timesteps} timesteps")
    print(f"State dim: {state_dim} (base: 11 + decisions: {args.history_k * 2} + payoffs: {args.history_k * 2} = {11 + args.history_k * 4})")
    print(f"History k: {args.history_k}")
    print(f"Max ep len: {max_ep_len}")
    print("=" * 50)
    
    num_folds = args.num_folds
    
    # Split data: 90% for 5-fold CV, 10% for test
    rng = np.random.default_rng(args.fold_seed)
    perm = rng.permutation(total_trajs)
    
    # Calculate split point: 90% for CV, 10% for test
    cv_size = int(total_trajs * 0.9)
    test_size = total_trajs - cv_size
    
    cv_inds = perm[:cv_size]  # 90% for cross-validation
    test_inds = perm[cv_size:]  # 10% for test set
    
    print("=" * 50)
    print(f"Data split: {len(cv_inds)} trajectories (90%) for {num_folds}-fold CV")
    print(f"            {len(test_inds)} trajectories (10%) for test set")
    print("=" * 50)
    
    # Create folds from CV data (90%)
    fold_sizes = np.full(num_folds, cv_size // num_folds, dtype=int)
    fold_sizes[: cv_size % num_folds] += 1
    folds, start = [], 0
    for size in fold_sizes:
        folds.append(cv_inds[start : start + size])
        start += size
    
    fold_val_accuracies: list[float] = []
    trained_models = []
    
    # Initialize wandb if requested
    if args.log_to_wandb:
        wandb_api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY")
        if not wandb_api_key:
            print("Warning: W&B logging requested but WANDB_API_KEY is not set. Skipping wandb logging.")
            args.log_to_wandb = False
        else:
            os.environ["WANDB_API_KEY"] = wandb_api_key
            wandb.login(key=wandb_api_key, relogin=True)
    
    def train_single_fold(fold_idx: int, train_inds: np.ndarray, val_inds: np.ndarray):
        print(f"\n--- Fold {fold_idx + 1}/{num_folds} ---")
        print(f"Train trajectories: {len(train_inds)} | Val trajectories: {len(val_inds)}")
        
        # Prepare state and action trajectories
        train_trajs = [trajectories[int(i)] for i in train_inds]
        val_trajs = [trajectories[int(i)] for i in val_inds]
        
        # Normalize states
        all_train_states = np.concatenate([tr['states'] for tr in train_trajs], axis=0)
        state_mean = all_train_states.mean(axis=0)
        state_std = all_train_states.std(axis=0) + 1e-6
        
        # Find max sequence length (K for context window)
        K = args.K if hasattr(args, 'K') else max(tr['states'].shape[0] for tr in train_trajs) if train_trajs else 40
        
        # Prepare training data: sliding windows
        train_data_samples = []
        for tr in train_trajs:
            T = tr['states'].shape[0]
            states = (tr['states'] - state_mean) / state_std
            actions = tr['actions']
            
            for start_idx in range(T):
                seq_len = min(K, T - start_idx)
                if seq_len <= 0:
                    continue
                
                s_window = states[start_idx:start_idx + seq_len]  # [seq_len, state_dim]
                a_window = actions[start_idx:start_idx + seq_len]  # [seq_len]
                
                # Pad to K if needed
                if seq_len < K:
                    s_pad = np.zeros((K - seq_len, state_dim), dtype=np.float32)
                    a_pad = np.zeros((K - seq_len,), dtype=np.float32)
                    s_window = np.concatenate([s_pad, s_window], axis=0)
                    a_window = np.concatenate([a_pad, a_window], axis=0)
                    mask = np.concatenate([np.zeros(K - seq_len), np.ones(seq_len)]).astype(np.float32)
                else:
                    mask = np.ones(seq_len, dtype=np.float32)
                
                train_data_samples.append((s_window, a_window, mask))
        
        if len(train_data_samples) == 0:
            raise ValueError("No training samples generated!")
        
        train_states_tensor = torch.tensor(np.stack([s for s, _, _ in train_data_samples]), dtype=torch.float32, device=device)
        train_actions_tensor = torch.tensor(np.stack([a for _, a, _ in train_data_samples]), dtype=torch.float32, device=device)
        train_masks_tensor = torch.tensor(np.stack([m for _, _, m in train_data_samples]), dtype=torch.long, device=device)
        
        # Create model
        n_nodes, n_layers = args.embed_dim, args.n_layer
        bert = bertModel(state_dim, n_nodes, n_layers, num_heads=args.n_head, dropout=args.dropout, max_len=max_ep_len).to(device)
        
        optimizer = torch.optim.AdamW(
            bert.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        
        loss_set_bert = []
        val_loss_set_bert = []
        
        # Initialize wandb for this fold
        wandb_run = None
        if args.log_to_wandb:
            group_name = "ipd-bert-v1"
            run_name = f"{group_name}-fold{fold_idx}-{random.randint(int(1e5), int(1e6) - 1)}"
            wandb_dir = os.path.join(os.path.dirname(__file__), "wandb")
            wandb_run = wandb.init(
                name=run_name,
                group=group_name,
                project="decision-transformer",
                config=vars(args) | {"fold_index": fold_idx},
                dir=wandb_dir,
            )
        
        print("Training BERT model...")
        n_epochs = args.max_iters
        batch_size = args.batch_size
        window = 10
        
        for ep in range(n_epochs):
            bert.train()
            epoch_losses = []
            epoch_accuracies = []
            
            # Shuffle training data
            indices = np.random.permutation(len(train_states_tensor))
            train_states_shuffled = train_states_tensor[indices]
            train_actions_shuffled = train_actions_tensor[indices]
            train_masks_shuffled = train_masks_tensor[indices]
            
            num_batches = len(train_states_tensor) // batch_size
            for bc in range(num_batches):
                start_idx = bc * batch_size
                end_idx = start_idx + batch_size
                
                states_batch = train_states_shuffled[start_idx:end_idx]  # [B, K, state_dim]
                actions_batch = train_actions_shuffled[start_idx:end_idx]  # [B, K]
                masks_batch = train_masks_shuffled[start_idx:end_idx]  # [B, K]
                
                # Forward pass
                logits = bert(states_batch, mask=masks_batch)  # [B, K, 1]
                
                # Flatten for loss computation
                logits_flat = logits.view(-1, 1)  # [B*K, 1]
                labels_flat = actions_batch.contiguous().view(-1, 1).float()  # [B*K, 1]
                mask_flat = (masks_batch.view(-1) == 1)
                
                if mask_flat.sum() == 0:
                    continue
                
                # Apply mask
                logits_masked = logits_flat[mask_flat]  # [N, 1]
                labels_masked = labels_flat[mask_flat]  # [N, 1]
                
                # Binary cross-entropy loss
                loss = F.binary_cross_entropy_with_logits(logits_masked, labels_masked, reduction='mean')
                
                # Compute accuracy
                probs = torch.sigmoid(logits_masked)
                preds = (probs >= 0.5).float()
                accuracy = (preds == labels_masked).float().mean().item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print_loss = loss.item()
                epoch_losses.append(print_loss)
                epoch_accuracies.append(accuracy)
                loss_set_bert.append(print_loss)
                
                if bc % window == 0:
                    print(f"Fold {fold_idx + 1} | Epoch[{ep + 1}/{n_epochs}], Batch[{bc + 1}/{num_batches}], Loss: {print_loss:.5f}, Acc: {accuracy:.3f}")
            
            # Validation step
            bert.eval()
            val_losses = []
            val_accuracies = []
            with torch.no_grad():
                for tr in val_trajs:
                    T = tr['states'].shape[0]
                    states = (tr['states'] - state_mean) / state_std
                    actions = tr['actions']
                    
                    # Process with sliding window
                    for start_idx in range(T):
                        seq_len = min(K, T - start_idx)
                        if seq_len <= 0:
                            continue
                        
                        s_window = states[start_idx:start_idx + seq_len]
                        a_window = actions[start_idx:start_idx + seq_len]
                        
                        # Pad to K if needed
                        if seq_len < K:
                            s_pad = np.zeros((K - seq_len, state_dim), dtype=np.float32)
                            a_pad = np.zeros((K - seq_len,), dtype=np.float32)
                            s_window = np.concatenate([s_pad, s_window], axis=0)
                            a_window = np.concatenate([a_pad, a_window], axis=0)
                            mask = np.concatenate([np.zeros(K - seq_len), np.ones(seq_len)]).astype(np.float32)
                        else:
                            mask = np.ones(seq_len, dtype=np.float32)
                        
                        states_tensor = torch.tensor(s_window, dtype=torch.float32, device=device).unsqueeze(0)
                        actions_tensor = torch.tensor(a_window, dtype=torch.float32, device=device).unsqueeze(0)
                        mask_tensor = torch.tensor(mask, dtype=torch.long, device=device).unsqueeze(0)
                        
                        logits = bert(states_tensor, mask=mask_tensor)
                        logits_flat = logits.view(-1, 1)
                        labels_flat = actions_tensor.contiguous().view(-1, 1).float()
                        mask_flat = (mask_tensor.view(-1) == 1)
                        
                        if mask_flat.sum() == 0:
                            continue
                        
                        logits_masked = logits_flat[mask_flat]
                        labels_masked = labels_flat[mask_flat]
                        
                        loss = F.binary_cross_entropy_with_logits(logits_masked, labels_masked, reduction='mean')
                        probs = torch.sigmoid(logits_masked)
                        preds = (probs >= 0.5).float()
                        accuracy = (preds == labels_masked).float().mean().item()
                        
                        val_losses.append(loss.item())
                        val_accuracies.append(accuracy)
            
            val_loss = np.mean(val_losses) if val_losses else 0.0
            val_accuracy = np.mean(val_accuracies) if val_accuracies else 0.0
            val_loss_set_bert.append(val_loss)
            
            train_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            train_accuracy = np.mean(epoch_accuracies) if epoch_accuracies else 0.0
            
            if ep % 2 == 0:
                print(f"BERT Epoch {ep+1}: Train Loss: {train_loss:.5f}, Train Acc: {train_accuracy:.3f}, Val Loss: {val_loss:.5f}, Val Acc: {val_accuracy:.3f}")
            
            # Log to wandb
            if wandb_run is not None:
                wandb.log({
                    "training/train_loss": train_loss,
                    "training/train_accuracy": train_accuracy,
                    "validation/val_loss": val_loss,
                    "validation/val_accuracy": val_accuracy,
                    "epoch": ep + 1,
                })
        
        bert = bert.eval()
        
        # Evaluate with IPD metrics
        print("\n" + "=" * 70)
        print(f"Evaluating IPD metrics for Fold {fold_idx + 1}/{num_folds}")
        print("=" * 70)
        metrics = evaluate_bert_ipd_metrics(
            model=bert,
            trajectories=val_trajs,
            state_mean=state_mean,
            state_std=state_std,
            state_dim=state_dim,
            device=device,
            K=K,
            max_ep_len=max_ep_len,
            print_report=True,
        )
        
        # Print formatted table
        print("\n" + "=" * 70)
        print(f"Fold {fold_idx + 1}/{num_folds} - Performance Metrics")
        print("=" * 70)
        print(f"{'Metric':<15} {'Value':>10}")
        print("-" * 70)
        print(f"{'Acc. t = 1':<15} {metrics['Acc.t=1']:>10.3f}")
        print(f"{'Acc. t > 1':<15} {metrics['Acc.t>1']:>10.3f}")
        print(f"{'LL t = 1':<15} {metrics['LL.t=1']:>10.0f}")
        print(f"{'LL t > 1':<15} {metrics['LL.t>1']:>10.0f}")
        print(f"{'Cor-Time':<15} {metrics['Cor-Time']:>10.3f}")
        print(f"{'Cor-Avg.':<15} {metrics['Cor-Avg']:>10.3f}")
        print(f"{'RMSE-Time':<15} {metrics['RMSE-Time']:>10.3f}")
        print(f"{'RMSE-Avg.':<15} {metrics['RMSE-Avg']:>10.3f}")
        print("=" * 70 + "\n")
        
        if wandb_run is not None:
            wandb.log({f"ipd/{k}": v for k, v in metrics.items()})
            wandb.finish()
        
        return val_accuracy, bert, state_mean, state_std
    
    # Train all folds
    for fold_idx in range(num_folds):
        val_inds = folds[fold_idx]
        train_inds = np.concatenate([folds[i] for i in range(num_folds) if i != fold_idx])
        
        val_accuracy, model, state_mean, state_std = train_single_fold(fold_idx, train_inds, val_inds)
        if val_accuracy is not None:
            fold_val_accuracies.append(val_accuracy)
        if model is not None:
            trained_models.append((model, state_mean, state_std))
    
    if fold_val_accuracies:
        avg_val_accuracy = float(np.mean(fold_val_accuracies))
        print("\n" + "=" * 50)
        print(f"Average validation accuracy over {len(fold_val_accuracies)} fold(s): {avg_val_accuracy:.6f}")
        print("=" * 50)
    
    # Evaluate on test set (10%)
    if len(test_inds) > 0 and len(trained_models) > 0:
        print("\n" + "=" * 70)
        print("Evaluating on Test Set (10% held-out)")
        print("=" * 70)
        
        # Use the model from the last fold for test evaluation
        test_model, test_state_mean, test_state_std = trained_models[-1]
        test_trajs = [trajectories[int(i)] for i in test_inds]
        
        # Evaluate on test set
        K = args.K if hasattr(args, 'K') else max(tr['states'].shape[0] for tr in test_trajs) if test_trajs else 40
        test_metrics = evaluate_bert_ipd_metrics(
            model=test_model,
            trajectories=test_trajs,
            state_mean=test_state_mean,
            state_std=test_state_std,
            state_dim=state_dim,
            device=device,
            K=K,
            max_ep_len=max_ep_len,
            print_report=True,
        )
        
        # Print test set results
        print("\n" + "=" * 70)
        print("Test Set (10% held-out) - Performance Metrics")
        print("=" * 70)
        print(f"{'Metric':<15} {'Value':>10}")
        print("-" * 70)
        print(f"{'Acc. t = 1':<15} {test_metrics['Acc.t=1']:>10.3f}")
        print(f"{'Acc. t > 1':<15} {test_metrics['Acc.t>1']:>10.3f}")
        print(f"{'LL t = 1':<15} {test_metrics['LL.t=1']:>10.0f}")
        print(f"{'LL t > 1':<15} {test_metrics['LL.t>1']:>10.0f}")
        print(f"{'Cor-Time':<15} {test_metrics['Cor-Time']:>10.3f}")
        print(f"{'Cor-Avg.':<15} {test_metrics['Cor-Avg']:>10.3f}")
        print(f"{'RMSE-Time':<15} {test_metrics['RMSE-Time']:>10.3f}")
        print(f"{'RMSE-Avg.':<15} {test_metrics['RMSE-Avg']:>10.3f}")
        print("=" * 70)
        
        # Log test metrics to wandb
        if args.log_to_wandb:
            wandb_api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY")
            if wandb_api_key:
                os.environ["WANDB_API_KEY"] = wandb_api_key
                wandb.login(key=wandb_api_key, relogin=True)
                group_name = "ipd-bert-v1"
                test_run_name = f"{group_name}-test-{random.randint(int(1e5), int(1e6) - 1)}"
                wandb_dir = os.path.dirname(__file__)
                test_wandb_run = wandb.init(
                    name=test_run_name,
                    group=group_name,
                    project="decision-transformer",
                    config=vars(args) | {"test_evaluation": True},
                    dir=wandb_dir,
                )
                # Log test metrics with "test/" prefix
                wandb.log({f"test/ipd/{k}": v for k, v in test_metrics.items()})
                wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="decision/data/all_data.csv")
    parser.add_argument("--history_k", type=int, default=3, help="Number of history decision columns to include in model (default: 3)")
    parser.add_argument("--K", type=int, default=40, help="Context length for BERT model")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128, help="Hidden dimension (n_nodes)")
    parser.add_argument("--n_layer", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--max_iters", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_to_wandb", action="store_true")
    parser.add_argument("--wandb_api_key", type=str, default="")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--fold_seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
