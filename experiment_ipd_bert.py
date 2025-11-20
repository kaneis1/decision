import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb

from transformers import BertConfig, BertModel
from typing import Dict, List
from decision_transformer.models.bert_decision_transformer import BertDecisionTransformer


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


def main(args):
    device = args.device
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # Load CSV -> trajectories (same as experiment_ipd2.py)
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

    # Evaluation function: compute same metrics as experiment_ipd2.py
    def evaluate_bert_ipd_metrics(
        model,
        trajectories: List[Dict],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        state_dim: int,
        device: torch.device,
        K: int,
        max_ep_len: int,
        rtg_scale: float,
        print_report: bool = True,
    ) -> Dict[str, float]:
        """Evaluate BERT Decision Transformer model with same metrics as Decision Transformer."""
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
                returns = traj['returns']  # [T] - RTG
                T = states.shape[0]
                
                # Normalize states
                states_norm = (states - state_mean) / state_std
                
                # Process trajectory with sliding window of size K (DT style)
                for start_idx in range(T):
                    seq_len = min(K, T - start_idx)
                    if seq_len <= 0:
                        continue
                    
                    # Extract sequence window
                    s_window = states_norm[start_idx:start_idx + seq_len]  # [seq_len, state_dim]
                    a_window = actions_true[start_idx:start_idx + seq_len]  # [seq_len]
                    r_window = returns[start_idx:start_idx + seq_len]  # [seq_len]
                    
                    # Compute RTG for this window
                    rtg_window = discount_cumsum(r_window, gamma=1.0).reshape(-1, 1) / rtg_scale  # [seq_len, 1]
                    
                    # Pad to K
                    tlen = seq_len
                    s_pad = np.zeros((K - tlen, state_dim), dtype=np.float32)
                    a_pad = np.zeros((K - tlen,), dtype=np.float32)
                    rtg_pad = np.zeros((K - tlen, 1), dtype=np.float32)
                    ts = np.arange(start_idx, start_idx + seq_len, dtype=np.int64)
                    ts[ts >= max_ep_len] = max_ep_len - 1
                    ts_pad = np.zeros((K - tlen,), dtype=np.int64)
                    
                    s_padded = np.concatenate([s_pad, s_window], axis=0).astype(np.float32)  # [K, state_dim]
                    a_padded = np.concatenate([a_pad, a_window], axis=0).astype(np.float32)  # [K]
                    rtg_padded = np.concatenate([rtg_pad, rtg_window], axis=0).astype(np.float32)  # [K, 1]
                    ts_padded = np.concatenate([ts_pad, ts], axis=0).astype(np.int64)  # [K]
                    mask = np.concatenate([np.zeros(K - tlen), np.ones(tlen)]).astype(np.float32)  # [K]
                    
                    # Convert to tensors
                    states_tensor = torch.tensor(s_padded, dtype=torch.float32, device=device).unsqueeze(0)  # [1, K, state_dim]
                    actions_tensor = torch.tensor(a_padded, dtype=torch.float32, device=device).unsqueeze(0)  # [1, K]
                    rtg_tensor = torch.tensor(rtg_padded, dtype=torch.float32, device=device).unsqueeze(0)  # [1, K, 1]
                    timesteps_tensor = torch.tensor(ts_padded, dtype=torch.long, device=device).unsqueeze(0)  # [1, K]
                    attention_mask = torch.tensor(mask, dtype=torch.long, device=device).unsqueeze(0)  # [1, K]
                    
                    # Get predictions using DT format
                    logits = model(
                        states=states_tensor,
                        actions=actions_tensor,
                        returns_to_go=rtg_tensor,
                        timesteps=timesteps_tensor,
                        attention_mask=attention_mask
                    )  # [1, K, 1]
                    
                    # Get prediction for the last timestep (current action)
                    if tlen > 0:
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
        cor_avg, rmse_avg = 0.0, 0.0  # Not applicable for GPT2 (no structure state)
        
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

    def train_single_fold(fold_idx: int, train_inds: np.ndarray, val_inds: np.ndarray) -> tuple[float | None, object]:
        
        print(f"\n--- Fold {fold_idx + 1}/{num_folds} ---")
        print(f"Train trajectories: {len(train_inds)} | Val trajectories: {len(val_inds)}")

        # Prepare state and action trajectories
        train_trajs = [trajectories[int(i)] for i in train_inds]
        val_trajs = [trajectories[int(i)] for i in val_inds]
        
        # Normalize states
        all_train_states = np.concatenate([tr['states'] for tr in train_trajs], axis=0)
        state_mean = all_train_states.mean(axis=0)
        state_std = all_train_states.std(axis=0) + 1e-6
        
        # Compute returns for scaling RTG
        all_train_returns = np.concatenate([tr['returns'] for tr in train_trajs], axis=0)
        rtg_scale = max(1.0, np.percentile(np.abs(all_train_returns), 95))
        
        # Find max sequence length (K for DT context window)
        K = args.K if hasattr(args, 'K') else max(tr['states'].shape[0] for tr in train_trajs) if train_trajs else 40
        
        # Prepare train data: states, actions, RTG, timesteps, masks (DT format)
        train_states_list = []
        train_actions_list = []
        train_rtg_list = []
        train_timesteps_list = []
        train_masks_list = []
        
        def make_padded_sample(traj, start_idx: int, K: int):
            T = traj['states'].shape[0]
            seq_len = min(K, T - start_idx)
            
            if seq_len <= 0:
                return None
                
            s_i = traj['states'][start_idx:start_idx + seq_len]  # [seq_len, state_dim]
            a_i = traj['actions'][start_idx:start_idx + seq_len]  # [seq_len]
            r_i = traj['rewards'][start_idx:start_idx + seq_len]  # [seq_len]
            returns_i = traj['returns'][start_idx:start_idx + seq_len]  # [seq_len] - RTG
            
            # Normalize states
            s_i = (s_i - state_mean) / state_std
            
            # Compute RTG (returns-to-go) - this is cumulative returns from current timestep
            rtg_i = discount_cumsum(r_i, gamma=1.0)  # [seq_len]
            rtg_i = rtg_i.reshape(-1, 1) / rtg_scale  # Normalize RTG
            
            # Timesteps: absolute timestep indices [0, 1, 2, ...]
            ts = np.arange(start_idx, start_idx + seq_len, dtype=np.int64)
            ts[ts >= max_ep_len] = max_ep_len - 1
            
            # Pad to K
            tlen = seq_len
            s_pad = np.zeros((K - tlen, state_dim), dtype=np.float32)
            a_pad = np.zeros((K - tlen,), dtype=np.float32)  # Use 0 for padding (will be masked)
            rtg_pad = np.zeros((K - tlen, 1), dtype=np.float32)
            ts_pad = np.zeros((K - tlen,), dtype=np.int64)
            
            s_i = np.concatenate([s_pad, s_i], axis=0).astype(np.float32)  # [K, state_dim]
            a_i = np.concatenate([a_pad, a_i], axis=0).astype(np.float32)  # [K]
            rtg_i = np.concatenate([rtg_pad, rtg_i], axis=0).astype(np.float32)  # [K, 1]
            ts = np.concatenate([ts_pad, ts], axis=0).astype(np.int64)  # [K]
            mask = np.concatenate([np.zeros(K - tlen), np.ones(tlen)]).astype(np.float32)  # [K]
            
            return s_i, a_i, rtg_i, ts, mask
        
        # Prepare training data
        train_data_samples = []
        for tr in train_trajs:
            T = tr['states'].shape[0]
            for si in range(T):
                sample = make_padded_sample(tr, si, K)
                if sample is not None:
                    train_data_samples.append(sample)
        
        if len(train_data_samples) == 0:
            raise ValueError("No training samples generated!")
        
        train_states_tensor = torch.tensor(np.stack([s for s, _, _, _, _ in train_data_samples]), dtype=torch.float32, device=device)  # [N, K, state_dim]
        train_actions_tensor = torch.tensor(np.stack([a for _, a, _, _, _ in train_data_samples]), dtype=torch.float32, device=device)  # [N, K]
        train_rtg_tensor = torch.tensor(np.stack([rtg for _, _, rtg, _, _ in train_data_samples]), dtype=torch.float32, device=device)  # [N, K, 1]
        train_timesteps_tensor = torch.tensor(np.stack([ts for _, _, _, ts, _ in train_data_samples]), dtype=torch.long, device=device)  # [N, K]
        train_masks_tensor = torch.tensor(np.stack([m for _, _, _, _, m in train_data_samples]), dtype=torch.float32, device=device)  # [N, K]
        
        # Prepare validation data (for evaluation)
        val_states_list = []
        val_actions_list = []
        val_rtg_list = []
        val_timesteps_list = []
        val_masks_list = []
        
        for tr in val_trajs:
            T = tr['states'].shape[0]
            states = (tr['states'] - state_mean) / state_std
            actions = tr['actions']
            returns = tr['returns']
            
            # Pad to K
            tlen = min(T, K)
            s_pad = np.zeros((K - tlen, state_dim), dtype=np.float32)
            a_pad = np.zeros((K - tlen,), dtype=np.float32)
            
            # Compute RTG for validation
            if tlen > 0:
                rtg_val = discount_cumsum(returns[:tlen], gamma=1.0).reshape(-1, 1) / rtg_scale
            else:
                rtg_val = np.zeros((tlen, 1), dtype=np.float32)
            rtg_pad = np.zeros((K - tlen, 1), dtype=np.float32)
            
            ts = np.arange(0, tlen, dtype=np.int64)
            ts[ts >= max_ep_len] = max_ep_len - 1
            ts_pad = np.zeros((K - tlen,), dtype=np.int64)
            
            states_padded = np.concatenate([s_pad, states[:tlen]], axis=0).astype(np.float32)
            actions_padded = np.concatenate([a_pad, actions[:tlen]], axis=0).astype(np.float32)
            rtg_padded = np.concatenate([rtg_pad, rtg_val], axis=0).astype(np.float32)
            ts_padded = np.concatenate([ts_pad, ts], axis=0).astype(np.int64)
            mask = np.concatenate([np.zeros(K - tlen), np.ones(tlen)]).astype(np.float32)
            
            val_states_list.append(states_padded)
            val_actions_list.append(actions_padded)
            val_rtg_list.append(rtg_padded)
            val_timesteps_list.append(ts_padded)
            val_masks_list.append(mask)
        
        val_states_tensor = torch.tensor(np.stack(val_states_list), dtype=torch.float32, device=device)
        val_actions_tensor = torch.tensor(np.stack(val_actions_list), dtype=torch.float32, device=device)
        val_rtg_tensor = torch.tensor(np.stack(val_rtg_list), dtype=torch.float32, device=device)
        val_timesteps_tensor = torch.tensor(np.stack(val_timesteps_list), dtype=torch.long, device=device)
        val_masks_tensor = torch.tensor(np.stack(val_masks_list), dtype=torch.float32, device=device)

        # Create model: BertDecisionTransformer
        bert_config = BertConfig(
            vocab_size=1,
            hidden_size=args.embed_dim,
            num_hidden_layers=args.n_layer,
            num_attention_heads=args.n_head,
            intermediate_size=4 * args.embed_dim,
            hidden_act=args.activation_function,
            hidden_dropout_prob=args.dropout,
            attention_probs_dropout_prob=args.dropout,
            max_position_embeddings=3 * K,  # 3K for interleaved (R, s, a) tokens
            type_vocab_size=1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
        )
        
        model = BertDecisionTransformer(
            state_dim=state_dim,
            act_dim=1,  # Binary classification
            hidden_size=args.embed_dim,
            K=K,
            max_ep_len=max_ep_len,
            bert_config=bert_config,
            use_action_ids=False,  # Actions are continuous (binary as float)
            action_tanh=False,  # Output raw logits for BCE loss
            binary_classification=True,  # Output 1 logit for binary classification
        ).to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps + 1) / args.warmup_steps, 1),
        )

        def get_batch(batch_size=args.batch_size):
            # Sample a batch from training data
            idxs = np.random.choice(len(train_states_tensor), size=batch_size, replace=True)
            states_batch = train_states_tensor[idxs]  # [B, K, state_dim]
            actions_batch = train_actions_tensor[idxs]  # [B, K]
            rtg_batch = train_rtg_tensor[idxs]  # [B, K, 1]
            timesteps_batch = train_timesteps_tensor[idxs]  # [B, K]
            masks_batch = train_masks_tensor[idxs]  # [B, K]
            return states_batch, actions_batch, rtg_batch, timesteps_batch, masks_batch

        def compute_val_loss():
            if val_inds.size == 0:
                return None

            was_training = model.training
            model.eval()

            total_loss = 0.0
            total_count = 0

            with torch.no_grad():
                # Use batches for validation
                val_batch_size = min(args.batch_size, len(val_states_tensor))
                for i in range(0, len(val_states_tensor), val_batch_size):
                    states_batch = val_states_tensor[i:i+val_batch_size]
                    actions_batch = val_actions_tensor[i:i+val_batch_size]
                    rtg_batch = val_rtg_tensor[i:i+val_batch_size]
                    timesteps_batch = val_timesteps_tensor[i:i+val_batch_size]
                    masks_batch = val_masks_tensor[i:i+val_batch_size]
                    
                    # Convert mask to BERT format: 1 for valid tokens, 0 for padding
                    attention_mask = masks_batch.long()  # [B, K] - BERT expects long type
                    
                    # DT format forward: states, actions, returns_to_go, timesteps, attention_mask
                    logits = model(
                        states=states_batch,
                        actions=actions_batch,  # [B, K] - used for past actions in context
                        returns_to_go=rtg_batch,  # [B, K, 1]
                        timesteps=timesteps_batch,  # [B, K]
                        attention_mask=attention_mask
                    )  # [B, K, 1]
                    
                    logits = logits.view(-1, 1)  # [B*K, 1]
                    labels = actions_batch.contiguous().view(-1, 1).float()  # [B*K, 1] - float for BCE
                    mask_flat = (masks_batch.view(-1) == 1)
                    
                    if mask_flat.sum() == 0:
                        continue
                    
                    # Apply mask to logits and labels
                    logits_masked = logits[mask_flat]  # [N, 1]
                    labels_masked = labels[mask_flat]  # [N, 1]
                    
                    loss = F.binary_cross_entropy_with_logits(logits_masked, labels_masked, reduction='none')
                    
                    total_loss += loss.sum().item()
                    total_count += loss.shape[0]

            if was_training:
                model.train()

            if total_count == 0:
                return None

            return total_loss / total_count

        def compute_val_accuracy():
            if val_inds.size == 0:
                return None

            was_training = model.training
            model.eval()

            total_correct = 0
            total_count = 0

            with torch.no_grad():
                # Use batches for validation
                val_batch_size = min(args.batch_size, len(val_states_tensor))
                for i in range(0, len(val_states_tensor), val_batch_size):
                    states_batch = val_states_tensor[i:i+val_batch_size]
                    actions_batch = val_actions_tensor[i:i+val_batch_size]
                    rtg_batch = val_rtg_tensor[i:i+val_batch_size]
                    timesteps_batch = val_timesteps_tensor[i:i+val_batch_size]
                    masks_batch = val_masks_tensor[i:i+val_batch_size]
                    
                    # Convert mask to BERT format: 1 for valid tokens, 0 for padding
                    attention_mask = masks_batch.long()  # [B, K] - BERT expects long type
                    
                    # DT format forward: states, actions, returns_to_go, timesteps, attention_mask
                    logits = model(
                        states=states_batch,
                        actions=actions_batch,  # [B, K] - used for past actions in context
                        returns_to_go=rtg_batch,  # [B, K, 1]
                        timesteps=timesteps_batch,  # [B, K]
                        attention_mask=attention_mask
                    )  # [B, K, 1]
                    
                    logits = logits.view(-1, 1)  # [B*K, 1]
                    labels = actions_batch.contiguous().view(-1, 1).float()  # [B*K, 1]
                    mask_flat = (masks_batch.view(-1) == 1)
                    
                    if mask_flat.sum() == 0:
                        continue
                    
                    # Apply mask to logits and labels
                    logits_masked = logits[mask_flat]  # [N, 1]
                    labels_masked = labels[mask_flat]  # [N, 1]
                    
                    # Apply sigmoid to get probabilities, then threshold at 0.5
                    action_probs = torch.sigmoid(logits_masked)  # [N, 1]
                    action_preds_binary = (action_probs >= 0.5).float()  # [N, 1]
                    
                    # Calculate number of correct predictions
                    correct = (action_preds_binary == labels_masked).float()  # [N, 1]
                    
                    total_correct += correct.sum().item()
                    total_count += correct.shape[0]

            if was_training:
                model.train()

            if total_count == 0:
                return None

            return total_correct / total_count

        wandb_run = None
        if args.log_to_wandb:
            wandb_api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY")
            if not wandb_api_key:
                raise RuntimeError("W&B logging requested but WANDB_API_KEY is not set.")
            os.environ["WANDB_API_KEY"] = wandb_api_key
            wandb.login(key=wandb_api_key, relogin=True)
            group_name = "ipd-bert"
            run_name = f"{group_name}-fold{fold_idx}-{random.randint(int(1e5), int(1e6) - 1)}"
            wandb_dir = os.path.join(os.path.dirname(__file__), "wandb")
            wandb_run = wandb.init(
                name=run_name,
                group=group_name,
                project="decision-transformer",
                config=vars(args) | {"fold_index": fold_idx},
                dir=wandb_dir,
            )

        last_val_accuracy = None
        for it in range(args.max_iters):
            itn = it + 1
            print(f">>> Fold {fold_idx + 1} | Iteration {itn}/{args.max_iters}")
            
            # Training
            model.train()
            total_train_loss = 0.0
            total_train_count = 0
            total_train_correct = 0
            total_train_samples = 0
            
            for step in range(args.num_steps_per_iter):
                states_batch, actions_batch, rtg_batch, timesteps_batch, masks_batch = get_batch()
                
                # Convert mask to BERT format: 1 for valid tokens, 0 for padding
                attention_mask = masks_batch.long()  # [B, K] - BERT expects long type
                
                # DT format forward: states, actions, returns_to_go, timesteps, attention_mask
                logits = model(
                    states=states_batch,
                    actions=actions_batch,  # [B, K] - used for past actions in context
                    returns_to_go=rtg_batch,  # [B, K, 1]
                    timesteps=timesteps_batch,  # [B, K]
                    attention_mask=attention_mask
                )  # [B, K, 1]
                
                logits = logits.view(-1, 1)  # [B*K, 1]
                labels = actions_batch.contiguous().view(-1, 1).float()  # [B*K, 1] - float for BCE
                mask_flat = (masks_batch.view(-1) == 1)
                
                if mask_flat.sum() == 0:
                    continue
                
                # Apply mask to logits and labels
                logits_masked = logits[mask_flat]  # [N, 1]
                labels_masked = labels[mask_flat]  # [N, 1]
                
                loss = F.binary_cross_entropy_with_logits(logits_masked, labels_masked, reduction='mean')
                
                # Compute training accuracy
                action_probs = torch.sigmoid(logits_masked)  # [N, 1]
                action_preds_binary = (action_probs >= 0.5).float()  # [N, 1]
                correct = (action_preds_binary == labels_masked).float()  # [N, 1]
                total_train_correct += correct.sum().item()
                total_train_samples += correct.shape[0]
                
                if loss.item() > 0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    total_train_loss += loss.item()
                    total_train_count += 1
            
            train_loss = total_train_loss / total_train_count if total_train_count > 0 else 0.0
            train_accuracy = total_train_correct / total_train_samples if total_train_samples > 0 else 0.0
            print(f"Training loss: {train_loss:.6f} | Training accuracy: {train_accuracy:.6f}")
            
            val_accuracy = compute_val_accuracy()
            if val_accuracy is not None:
                last_val_accuracy = val_accuracy
                print(f"Validation Accuracy: {val_accuracy:.6f}")
            
            logs = {
                "training/train_loss": train_loss,
                "training/train_accuracy": train_accuracy,
            }
            if val_accuracy is not None:
                logs["validation/accuracy"] = val_accuracy
            
            if args.log_to_wandb and wandb_run is not None:
                wandb.log(logs)

        print("\n" + "=" * 70)
        print(f"Evaluating IPD metrics for Fold {fold_idx + 1}/{num_folds}")
        print("=" * 70)
        metrics = evaluate_bert_ipd_metrics(
            model=model,
            trajectories=val_trajs,
            state_mean=state_mean,
            state_std=state_std,
            state_dim=state_dim,
            device=device,
            K=K,
            max_ep_len=max_ep_len,
            rtg_scale=rtg_scale,
            print_report=True,
        )
        
        # Print formatted table (same format as experiment_ipd2.py, excluding MO parts)
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
        
        if wandb_run is not None:
            wandb.finish()

        return last_val_accuracy, model

    # Store models from each fold for test evaluation
    trained_models = []
    
    for fold_idx in range(num_folds):
        val_inds = folds[fold_idx]
        train_inds = np.concatenate([folds[i] for i in range(num_folds) if i != fold_idx])

        val_accuracy, model = train_single_fold(fold_idx, train_inds, val_inds)
        if val_accuracy is not None:
            fold_val_accuracies.append(val_accuracy)
        if model is not None:
            trained_models.append(model)

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
        test_model = trained_models[-1]
        test_trajs = [trajectories[int(i)] for i in test_inds]
        
        # Calculate normalization stats from all CV data for test evaluation
        all_cv_states = np.concatenate([trajectories[int(i)]['states'] for i in cv_inds], axis=0)
        test_state_mean = all_cv_states.mean(axis=0)
        test_state_std = all_cv_states.std(axis=0) + 1e-6
        
        # Compute RTG scale from CV data
        all_cv_returns = np.concatenate([trajectories[int(i)]['returns'] for i in cv_inds], axis=0)
        test_rtg_scale = max(1.0, np.percentile(np.abs(all_cv_returns), 95))
        
        # Evaluate on test set
        test_metrics = evaluate_bert_ipd_metrics(
            model=test_model,
            trajectories=test_trajs,
            state_mean=test_state_mean,
            state_std=test_state_std,
            state_dim=state_dim,
            device=device,
            K=args.K,
            max_ep_len=max_ep_len,
            rtg_scale=test_rtg_scale,
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
        
        # Log test metrics to wandb (create a new run for test results)
        if args.log_to_wandb:
            wandb_api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY")
            if wandb_api_key:
                os.environ["WANDB_API_KEY"] = wandb_api_key
                wandb.login(key=wandb_api_key, relogin=True)
                group_name = "ipd-bert"
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
    parser.add_argument("--history_k", type=int, default=3, help="Number of history decision columns to include in model (default: 3). When history_k=3, includes my.decision1-3 and other.decision1-3")
    parser.add_argument("--K", type=int, default=40, help="Context length for Decision Transformer")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=15)
    parser.add_argument("--num_steps_per_iter", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_to_wandb", action="store_true")
    parser.add_argument("--wandb_api_key", type=str, default="")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--fold_index", type=int, default=0)
    parser.add_argument("--fold_seed", type=int, default=0)
    parser.add_argument("--run_all_folds", action="store_true", default=True)
    args = parser.parse_args()
    main(args)
