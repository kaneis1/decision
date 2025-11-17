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

# Use the same trajectory loading as experiment_ipd2.py
STATE_COLS_DEFAULT = [
    "risk", "error", "delta", "infin", "contin", "r1", "r2", "r", "s", "t", "p","my.decision1","other.decision1"
]

MODUS_OPERANDI_COLS_DEFAULT = [
    "r1","r2","risk","error","delta","r1*delta","r2*delta","infin","contin","delta*infin","my.decision1","other.decision1"
]

def load_ipd_csv_as_trajectories(csv_path: str):
    """Same as experiment_ipd2.py - loads full trajectories."""
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]

    period_col = "period"
    action_col = "my.decision"

    state_cols = STATE_COLS_DEFAULT
    modus_operandi_cols = MODUS_OPERANDI_COLS_DEFAULT

    # Compute interaction terms that don't exist in the CSV
    df["r1*delta"] = pd.to_numeric(df["r1"], errors="coerce").fillna(0.0) * pd.to_numeric(df["delta"], errors="coerce").fillna(0.0)
    df["r2*delta"] = pd.to_numeric(df["r2"], errors="coerce").fillna(0.0) * pd.to_numeric(df["delta"], errors="coerce").fillna(0.0)
    df["delta*infin"] = pd.to_numeric(df["delta"], errors="coerce").fillna(0.0) * pd.to_numeric(df["infin"], errors="coerce").fillna(0.0)

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

        #we set first period's decision to 2 because it don't have other.decision1 and my.decision1
        ep["my.decision1"] = ep["my.decision1"].fillna(2)
        ep["other.decision1"] = ep["other.decision1"].fillna(2)
        state_vals = (
            ep[state_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .astype(float)
            .values
        )
        modus_operandi_vals = (
            ep[modus_operandi_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .astype(float)
            .values
        )
        s = state_vals.astype(np.float32)
        a = ep[action_col].values.astype(np.float32).reshape(-1, 1)
        r = ep["my.payoff1"].fillna(0.0).values.astype(np.float32)
        mo = modus_operandi_vals.astype(np.float32)

        T = len(ep)
        terminals = np.zeros(T, dtype=np.int64)
        terminals[-1] = 1

        trajectories.append(dict(observations=s, actions=a, rewards=r, modus_operandi=mo, terminals=terminals, lens=int(T)))
        max_ep_len = max(max_ep_len, T)

    return trajectories, max_ep_len


def extract_decision_history_from_trajectories(trajectories: List[Dict], history_k: int = 5):
    """
    Extract decision history from full trajectories for GPT2.
    
    Uses the same data source as experiment_ipd2.py - extracts decision history
    from observations which contain my.decision1, other.decision1, etc.
    
    Args:
        trajectories: List of trajectory dicts from load_ipd_csv_as_trajectories
        history_k: Number of history steps to include
    
    Returns:
        gpt2_trajectories: List of dict with 'decisions' key (np.ndarray of shape [T, n_history_feats])
        max_ep_len: int
    """
    gpt2_trajectories = []
    max_ep_len = 1
    
    for traj in trajectories:
        observations = traj["observations"]  # [T, state_dim]
        T = observations.shape[0]
        
        # STATE_COLS_DEFAULT = ["risk", "error", "delta", "infin", "contin", "r1", "r2", "r", "s", "t", "p", "my.decision1", "other.decision1"]
        # So my.decision1 is at index 11, other.decision1 is at index 12
        # We extract decision history columns: my.decision1, other.decision1, my.decision2, other.decision2, ...
        
        # Build decision history array
        decision_history = []
        for t in range(T):
            step = []
            # Extract my.decision1 and other.decision1 (indices 11, 12)
            if observations.shape[1] > 12:
                my_dec1 = observations[t, 11]
                other_dec1 = observations[t, 12]
                step.extend([float(my_dec1), float(other_dec1)])
                
                # Extract additional history if available in observations
                # Check if there are more decision columns beyond index 12
                # For history_k > 1, we need my.decision2, other.decision2, etc.
                # These should be in additional columns if they exist in the CSV
                # For now, we use what's available (at least decision1)
                # If more history is needed, it should be added to STATE_COLS_DEFAULT
            else:
                # Fallback: use actions from trajectory if decision columns not available
                my_action = float(traj["actions"][t, 0])
                # For other.decision, we don't have it directly, so use 0 or try to infer
                step.extend([my_action, 0.0])
            
            decision_history.append(step)
        
        decision_history = np.array(decision_history, dtype=np.float32)  # [T, n_history_feats]
        max_ep_len = max(max_ep_len, T)
        gpt2_trajectories.append({'decisions': decision_history, 'lens': T})
    
    return gpt2_trajectories, max_ep_len


def main(args):
    device = args.device
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # Load CSV -> trajectories (same as experiment_ipd2.py)
    full_trajectories, max_ep_len = load_ipd_csv_as_trajectories(
        csv_path=args.csv_path,
    )
    
    # Extract decision history from full trajectories for GPT2
    trajectories, max_ep_len = extract_decision_history_from_trajectories(
        full_trajectories,
        history_k=args.history_k,
    )

    histories = np.concatenate([tr['decisions'] for tr in trajectories], axis=0)
    n_history_feats = histories.shape[1]
    traj_lens = np.array([tr["lens"] for tr in trajectories], dtype=np.int32)
    num_timesteps = int(traj_lens.sum())
    total_trajs = len(trajectories)

    print("=" * 50)
    print(f"IPD-CSV dataset (GPT2): {total_trajs} trajectories, {num_timesteps} timesteps")
    print(f"History features: {n_history_feats}, Max ep len: {max_ep_len}")
    print("=" * 50)

    num_folds = args.num_folds

    rng = np.random.default_rng(args.fold_seed)
    perm = rng.permutation(total_trajs)
    fold_sizes = np.full(num_folds, total_trajs // num_folds, dtype=int)
    fold_sizes[: total_trajs % num_folds] += 1
    folds, start = [], 0
    for size in fold_sizes:
        folds.append(perm[start : start + size])
        start += size

    fold_val_losses: list[float] = []

    def train_single_fold(fold_idx: int, train_inds: np.ndarray, val_inds: np.ndarray) -> float | None:
        
        print(f"\n--- Fold {fold_idx + 1}/{num_folds} ---")
        print(f"Train trajectories: {len(train_inds)} | Val trajectories: {len(val_inds)}")

        # Prepare tokenized trajectories for train set
        train_trajs = [trajectories[int(i)] for i in train_inds]
        val_trajs = [trajectories[int(i)] for i in val_inds]
        
        tokenized_train = []
        for tr in train_trajs:
            tokens = np.ravel(tr['decisions'].astype(np.int64))  # [T * n_history_feats]
            tokenized_train.append(tokens)
        
        # Find max sequence length
        seq_length = max(len(t) for t in tokenized_train) if tokenized_train else 1
        
        # Pad all train trajectories to seq_length
        train_input_ids = np.full((len(tokenized_train), seq_length), fill_value=-100, dtype=np.int64)
        for i, t in enumerate(tokenized_train):
            train_input_ids[i, :len(t)] = t
        
        train_input_ids_tensor = torch.tensor(train_input_ids, dtype=torch.long, device=device)
        
        # Prepare validation trajectories
        tokenized_val = []
        for tr in val_trajs:
            tokens = np.ravel(tr['decisions'].astype(np.int64))
            tokenized_val.append(tokens)
        
        val_input_ids = np.full((len(tokenized_val), seq_length), fill_value=-100, dtype=np.int64)
        for i, t in enumerate(tokenized_val):
            val_input_ids[i, :len(t)] = t
        
        val_input_ids_tensor = torch.tensor(val_input_ids, dtype=torch.long, device=device)

        # GPT2Config for our tiny-vocab, pure decision history
        config = GPT2Config(
            vocab_size=3,  # 0, 1, (maybe pad/ignore)
            n_positions=seq_length,
            n_embd=args.embed_dim,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_inner=4 * args.embed_dim,
            activation_function=args.activation_function,
            resid_pdrop=args.dropout,
            attn_pdrop=args.dropout,
            add_cross_attention=False,
            use_cache=False,
        )
        model = GPT2Model(config).to(device)
        
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
            idxs = np.random.choice(len(train_input_ids_tensor), size=batch_size, replace=True)
            batch = train_input_ids_tensor[idxs]
            # Inputs: except last, Targets: except first
            input_seq = batch[:, :-1]
            target_seq = batch[:, 1:]
            # Attention mask: ignore padding
            mask = (input_seq != -100).long()
            input_seq[input_seq == -100] = 0  # GPT2 can't take negative input_ids
            return input_seq, target_seq, mask

        def compute_val_loss():
            if val_inds.size == 0:
                return None

            was_training = model.training
            model.eval()

            total_loss = 0.0
            total_count = 0

            with torch.no_grad():
                # Use batches for validation
                val_batch_size = min(args.batch_size, len(val_input_ids_tensor))
                for i in range(0, len(val_input_ids_tensor), val_batch_size):
                    batch = val_input_ids_tensor[i:i+val_batch_size]
                    input_seq = batch[:, :-1]
                    target_seq = batch[:, 1:]
                    mask = (input_seq != -100).long()
                    input_seq[input_seq == -100] = 0

                    outputs = model(input_ids=input_seq)
                    logits = outputs.last_hidden_state
                    logits = logits.view(-1, logits.size(-1))
                    labels = target_seq.contiguous().view(-1)
                    mask_flat = (mask.view(-1) == 1)
                    
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
                input_seq, target_seq, mask = get_batch()
                outputs = model(input_ids=input_seq)
                logits = outputs.last_hidden_state
                logits = logits.view(-1, logits.size(-1))
                labels = target_seq.contiguous().view(-1)
                mask_flat = (mask.view(-1) == 1)
                
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

        # Evaluation: compute same metrics as experiment_ipd2.py
        def evaluate_gpt2_ipd_metrics(
            model,
            trajectories: List[Dict],
            n_history_feats: int,
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
                    decisions = traj['decisions']  # [T, n_history_feats]
                    T = decisions.shape[0]
                    
                    # Get ground truth actions (first element of each timestep)
                    actions_true = decisions[:, 0].astype(np.float32)  # [T]
                    
                    # Get tokenized sequence for this trajectory
                    tokens_flat = np.ravel(decisions.astype(np.int64))  # [T * n_history_feats]
                    
                    # Evaluate at each timestep
                    for si in range(T - 1):  # Can't predict last timestep
                        # We want to predict my.decision at timestep si+1
                        # Input: all tokens up to (but not including) the first token of timestep si+1
                        # The first token of timestep si+1 is at position (si+1) * n_history_feats
                        input_end_pos = (si + 1) * n_history_feats
                        if input_end_pos == 0:
                            continue
                        
                        # Input sequence: all tokens before the token we want to predict
                        input_seq = tokens_flat[:input_end_pos].copy()
                        if len(input_seq) == 0:
                            continue
                        
                        # Convert to tensor
                        input_t = torch.tensor(input_seq, dtype=torch.long, device=device).unsqueeze(0)
                        
                        # Model forward - predict next token
                        outputs = model(input_ids=input_t)
                        # Get logits for the last position (predicting the token at input_end_pos)
                        logits = outputs.last_hidden_state[0, -1]  # Logits for last position
                        
                        # The token we're predicting is the first token of timestep si+1
                        # which is my.decision at timestep si+1
                        target_token = tokens_flat[input_end_pos]
                        
                        # Get probability of action=1 (token=1)
                        probs = F.softmax(logits, dim=-1)
                        p_action1 = float(probs[1].item()) if len(probs) > 1 else 0.0
                        y_true = float(actions_true[si + 1])
                        
                        # Store predictions
                        t_abs = si + 1  # 1-based absolute time (si=0 means predicting t=1)
                        if t_abs == 1:
                            y_t1.append(y_true)
                            p_t1.append(p_action1)
                        else:
                            y_tgt.append(y_true)
                            p_tgt.append(p_action1)
                        
                        per_time_truth.setdefault(t_abs, []).append(y_true)
                        per_time_pred.setdefault(t_abs, []).append(p_action1)
            
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
        
        print("\n" + "=" * 70)
        print(f"Evaluating IPD metrics for Fold {fold_idx + 1}/{num_folds}")
        print("=" * 70)
        metrics = evaluate_gpt2_ipd_metrics(
            model=model,
            trajectories=val_trajs,
            n_history_feats=n_history_feats,
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

        return last_val_loss

    for fold_idx in range(num_folds):
        val_inds = folds[fold_idx]
        train_inds = np.concatenate([folds[i] for i in range(num_folds) if i != fold_idx])

        val_loss = train_single_fold(fold_idx, train_inds, val_inds)
        if val_loss is not None:
            fold_val_losses.append(val_loss)

    if fold_val_losses:
        avg_val_loss = float(np.mean(fold_val_losses))
        print("\n" + "=" * 50)
        print(f"Average validation loss over {len(fold_val_losses)} fold(s): {avg_val_loss:.6f}")
        print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data/all_data.csv")
    parser.add_argument("--history_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=10)
    parser.add_argument("--num_steps_per_iter", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_to_wandb", action="store_true")
    parser.add_argument("--wandb_api_key", type=str, default="")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--fold_index", type=int, default=0)
    parser.add_argument("--fold_seed", type=int, default=0)
    parser.add_argument("--run_all_folds", action="store_true", default=True)
    args = parser.parse_args()
    main(args)
