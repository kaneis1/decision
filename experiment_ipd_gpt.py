import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb

from decision_transformer.models.trajectory_gpt2 import GPT2Config, GPT2Model
from typing import Dict, List


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

        T = len(ep)
        terminals = np.zeros(T, dtype=np.int64)
        terminals[-1] = 1

        trajectories.append(dict(observations=s, actions=a, terminals=terminals, lens=int(T)))
        max_ep_len = max(max_ep_len, T)

    return trajectories, max_ep_len


def extract_state_from_trajectories(trajectories: List[Dict], history_k: int = 1):
    """
    Extract state from full trajectories for GPT2.
    
    State includes:
    - Base features: "risk", "error", "delta", "infin", "contin", "r1", "r2", "r", "s", "t", "p" (11 features)
    - Decision history: my.decision1, other.decision1, my.decision2, other.decision2, ... (history_k * 2 features)
    - Payoff history: my.payoff1, other.payoff1, my.payoff2, other.payoff2, ... (history_k * 2 features)
    
    Args:
        trajectories: List of trajectory dicts from load_ipd_csv_as_trajectories
        history_k: Number of history steps to include
    
    Returns:
        gpt2_trajectories: List of dict with 'states' key (np.ndarray of shape [T, state_dim]) and 'actions' key
        max_ep_len: int
    """
    gpt2_trajectories = []
    max_ep_len = 1
    
    # State structure: 
    # - Base (11): ["risk", "error", "delta", "infin", "contin", "r1", "r2", "r", "s", "t", "p"]
    # - Decisions (history_k * 2): my.decision1, other.decision1, my.decision2, other.decision2, ...
    # - Payoffs (history_k * 2): my.payoff1, other.payoff1, my.payoff2, other.payoff2, ...
    # Total: 11 + history_k * 2 + history_k * 2 = 11 + history_k * 4
    
    for traj in trajectories:
        observations = traj["observations"]  # [T, state_dim] - already includes base + decisions + payoffs
        actions = traj["actions"]  # [T, 1]
        T = observations.shape[0]
        
        # States are already in the correct format from load_ipd_csv_as_trajectories
        states = observations.astype(np.float32)  # [T, state_dim]
        actions_flat = actions.reshape(-1).astype(np.float32)  # [T]
        
        max_ep_len = max(max_ep_len, T)
        gpt2_trajectories.append({
            'states': states, 
            'actions': actions_flat,
            'lens': T
        })
    
    return gpt2_trajectories, max_ep_len


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
    
    # Extract state from full trajectories for GPT2
    trajectories, max_ep_len = extract_state_from_trajectories(
        full_trajectories,
        history_k=args.history_k,
    )

    states = np.concatenate([tr['states'] for tr in trajectories], axis=0)
    state_dim = states.shape[1]
    traj_lens = np.array([tr["lens"] for tr in trajectories], dtype=np.int32)
    num_timesteps = int(traj_lens.sum())
    total_trajs = len(trajectories)

    print("=" * 50)
    print(f"IPD-CSV dataset (GPT2): {total_trajs} trajectories, {num_timesteps} timesteps")
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

    fold_val_losses: list[float] = []

    # Evaluation function: compute same metrics as experiment_ipd2.py
    def evaluate_gpt2_ipd_metrics(
        model,
        trajectories: List[Dict],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        state_dim: int,
        device: torch.device,
        print_report: bool = True,
    ) -> Dict[str, float]:
        """Evaluate GPT2 model with same metrics as Decision Transformer."""
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
                
                # Process entire trajectory at once
                states_tensor = torch.tensor(states_norm, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, state_dim]
                attention_mask = torch.ones((1, T), dtype=torch.float32, device=device)
                attention_mask_4d = attention_mask.unsqueeze(1).unsqueeze(2)  # [1, 1, 1, T]
                attention_mask_4d = (1.0 - attention_mask_4d) * -10000.0
                
                # Get predictions for all timesteps
                logits = model(states_tensor, attention_mask=attention_mask_4d)  # [1, T, 2]
                probs = F.softmax(logits, dim=-1)  # [1, T, 2]
                p_action1 = probs[0, :, 1].cpu().numpy()  # [T] - probability of action=1
                
                # Evaluate at each timestep
                for t_idx in range(T):
                    y_true = float(actions_true[t_idx])
                    p_pred = float(p_action1[t_idx])
                    
                    # Store predictions
                    t_abs = t_idx + 1  # 1-based absolute time
                    if t_abs == 1:
                        y_t1.append(y_true)
                        p_t1.append(p_pred)
                    else:
                        y_tgt.append(y_true)
                        p_tgt.append(p_pred)
                    
                    per_time_truth.setdefault(t_abs, []).append(y_true)
                    per_time_pred.setdefault(t_abs, []).append(p_pred)
        
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
            print("GPT2 IPD Eval →", metrics_str)
        
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
        
        # Find max sequence length
        seq_length = max(tr['states'].shape[0] for tr in train_trajs) if train_trajs else 1
        
        # Prepare train data: states and actions
        train_states_list = []
        train_actions_list = []
        train_masks_list = []
        
        for tr in train_trajs:
            T = tr['states'].shape[0]
            states = (tr['states'] - state_mean) / state_std  # Normalize
            actions = tr['actions']  # [T]
            
            # Pad to seq_length
            state_pad = np.zeros((seq_length - T, state_dim), dtype=np.float32)
            states_padded = np.concatenate([state_pad, states], axis=0)  # [seq_length, state_dim]
            
            action_pad = np.full((seq_length - T,), -100, dtype=np.float32)
            actions_padded = np.concatenate([action_pad, actions], axis=0)  # [seq_length]
            
            mask = np.concatenate([np.zeros(seq_length - T), np.ones(T)]).astype(np.float32)
            
            train_states_list.append(states_padded)
            train_actions_list.append(actions_padded)
            train_masks_list.append(mask)
        
        train_states_tensor = torch.tensor(np.stack(train_states_list), dtype=torch.float32, device=device)  # [N, seq_length, state_dim]
        train_actions_tensor = torch.tensor(np.stack(train_actions_list), dtype=torch.long, device=device)  # [N, seq_length]
        train_masks_tensor = torch.tensor(np.stack(train_masks_list), dtype=torch.float32, device=device)  # [N, seq_length]
        
        # Prepare validation data
        val_states_list = []
        val_actions_list = []
        val_masks_list = []
        
        for tr in val_trajs:
            T = tr['states'].shape[0]
            states = (tr['states'] - state_mean) / state_std
            actions = tr['actions']
            
            state_pad = np.zeros((seq_length - T, state_dim), dtype=np.float32)
            states_padded = np.concatenate([state_pad, states], axis=0)
            
            action_pad = np.full((seq_length - T,), -100, dtype=np.float32)
            actions_padded = np.concatenate([action_pad, actions], axis=0)
            
            mask = np.concatenate([np.zeros(seq_length - T), np.ones(T)]).astype(np.float32)
            
            val_states_list.append(states_padded)
            val_actions_list.append(actions_padded)
            val_masks_list.append(mask)
        
        val_states_tensor = torch.tensor(np.stack(val_states_list), dtype=torch.float32, device=device)
        val_actions_tensor = torch.tensor(np.stack(val_actions_list), dtype=torch.long, device=device)
        val_masks_tensor = torch.tensor(np.stack(val_masks_list), dtype=torch.float32, device=device)

        # Create model: State embedding + GPT2 + Action prediction head
        class StateGPT2Model(torch.nn.Module):
            def __init__(self, state_dim, embed_dim, n_layer, n_head, n_inner, activation_function, dropout, n_positions):
                super().__init__()
                self.state_embed = torch.nn.Linear(state_dim, embed_dim)
                self.gpt2 = GPT2Model(GPT2Config(
                    vocab_size=1,  # Dummy, we use inputs_embeds
                    n_positions=n_positions,
                    n_embd=embed_dim,
                    n_layer=n_layer,
                    n_head=n_head,
                    n_inner=n_inner,
                    activation_function=activation_function,
                    resid_pdrop=dropout,
                    attn_pdrop=dropout,
                    add_cross_attention=False,
                    use_cache=False,
                ))
                self.action_head = torch.nn.Linear(embed_dim, 2)  # Binary classification: 0 or 1
            
            def forward(self, states, attention_mask=None):
                # states: [B, T, state_dim]
                # Embed states
                inputs_embeds = self.state_embed(states)  # [B, T, embed_dim]
                # Process with GPT2
                outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
                # Predict actions
                logits = self.action_head(outputs.last_hidden_state)  # [B, T, 2]
                return logits
        
        model = StateGPT2Model(
            state_dim=state_dim,
            embed_dim=args.embed_dim,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_inner=4 * args.embed_dim,
            activation_function=args.activation_function,
            dropout=args.dropout,
            n_positions=seq_length,
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
            # Sample a batch of trajectories from training set
            idxs = np.random.choice(len(train_states_tensor), size=batch_size, replace=True)
            states_batch = train_states_tensor[idxs]  # [B, T, state_dim]
            actions_batch = train_actions_tensor[idxs]  # [B, T]
            masks_batch = train_masks_tensor[idxs]  # [B, T]
            return states_batch, actions_batch, masks_batch

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
                    masks_batch = val_masks_tensor[i:i+val_batch_size]
                    
                    attention_mask = masks_batch.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
                    attention_mask = (1.0 - attention_mask) * -10000.0
                    
                    logits = model(states_batch, attention_mask=attention_mask)  # [B, T, 2]
                    logits = logits.view(-1, 2)  # [B*T, 2]
                    labels = actions_batch.contiguous().view(-1)  # [B*T]
                    mask_flat = (masks_batch.view(-1) == 1)
                    
                    if mask_flat.sum() == 0:
                        continue
                    
                    loss = F.cross_entropy(logits, labels, ignore_index=-100, reduction='none')
                    loss = loss[mask_flat]
                    
                    total_loss += loss.sum().item()
                    total_count += loss.shape[0]

            if was_training:
                model.train()

            if total_count == 0:
                return None

            return total_loss / total_count

        wandb_run = None
        if args.log_to_wandb:
            wandb_api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY")
            if not wandb_api_key:
                raise RuntimeError("W&B logging requested but WANDB_API_KEY is not set.")
            os.environ["WANDB_API_KEY"] = wandb_api_key
            wandb.login(key=wandb_api_key, relogin=True)
            group_name = "ipd-gpt2"
            run_name = f"{group_name}-fold{fold_idx}-{random.randint(int(1e5), int(1e6) - 1)}"
            wandb_run = wandb.init(
                name=run_name,
                group=group_name,
                project="decision-transformer",
                config=vars(args) | {"fold_index": fold_idx},
            )

        last_val_loss = None
        for it in range(args.max_iters):
            itn = it + 1
            print(f">>> Fold {fold_idx + 1} | Iteration {itn}/{args.max_iters}")
            
            # Training
            model.train()
            total_train_loss = 0.0
            total_train_count = 0
            
            for step in range(args.num_steps_per_iter):
                states_batch, actions_batch, masks_batch = get_batch()
                
                attention_mask = masks_batch.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
                attention_mask = (1.0 - attention_mask) * -10000.0
                
                logits = model(states_batch, attention_mask=attention_mask)  # [B, T, 2]
                logits = logits.view(-1, 2)  # [B*T, 2]
                labels = actions_batch.contiguous().view(-1)  # [B*T]
                mask_flat = (masks_batch.view(-1) == 1)
                
                if mask_flat.sum() == 0:
                    continue
                
                loss = F.cross_entropy(logits, labels, ignore_index=-100, reduction='none')
                loss = loss[mask_flat]
                
                if loss.shape[0] > 0:
                    loss_mean = loss.mean()
                    optimizer.zero_grad()
                    loss_mean.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    total_train_loss += loss_mean.item()
                    total_train_count += 1
            
            train_loss = total_train_loss / total_train_count if total_train_count > 0 else 0.0
            print(f"Training loss: {train_loss:.6f}")
            
            val_loss = compute_val_loss()
            if val_loss is not None:
                last_val_loss = val_loss
                print(f"Validation loss: {val_loss:.6f}")
            
            logs = {
                "training/train_loss": train_loss,
            }
            if val_loss is not None:
                logs["validation/loss"] = val_loss
            
            if args.log_to_wandb and wandb_run is not None:
                wandb.log(logs)

        print("\n" + "=" * 70)
        print(f"Evaluating IPD metrics for Fold {fold_idx + 1}/{num_folds}")
        print("=" * 70)
        metrics = evaluate_gpt2_ipd_metrics(
            model=model,
            trajectories=val_trajs,
            state_mean=state_mean,
            state_std=state_std,
            state_dim=state_dim,
            device=device,
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

        return last_val_loss, model

    # Store models from each fold for test evaluation
    trained_models = []
    
    for fold_idx in range(num_folds):
        val_inds = folds[fold_idx]
        train_inds = np.concatenate([folds[i] for i in range(num_folds) if i != fold_idx])

        val_loss, model = train_single_fold(fold_idx, train_inds, val_inds)
        if val_loss is not None:
            fold_val_losses.append(val_loss)
        if model is not None:
            trained_models.append(model)

    if fold_val_losses:
        avg_val_loss = float(np.mean(fold_val_losses))
        print("\n" + "=" * 50)
        print(f"Average validation loss over {len(fold_val_losses)} fold(s): {avg_val_loss:.6f}")
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
        
        # Evaluate on test set
        test_metrics = evaluate_gpt2_ipd_metrics(
            model=test_model,
            trajectories=test_trajs,
            state_mean=test_state_mean,
            state_std=test_state_std,
            state_dim=state_dim,
            device=device,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="decision/data/all_data.csv")
    parser.add_argument("--history_k", type=int, default=3, help="Number of history decision columns to include in model (default: 5). When history_k=3, includes my.decision1-3 and other.decision1-3, resulting in 6 features per timestep")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=30)
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
