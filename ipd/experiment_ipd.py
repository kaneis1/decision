# experiment_ipd_csv.py
# Train a Decision Transformer on Iterated Prisoner's Dilemma CSV (all_data.csv)
#
# This mirrors experiment.py's structure but replaces the D4RL pkl loader
# with a CSV -> trajectories builder tailored for the IPD columns.

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import wandb

from decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.evaluation.evaluate_ipd_no_mo import evaluate_ipd_metrics


def discount_cumsum(x, gamma: float):
    """Same helper as experiment.py (gamma=1. for RTG)."""
    out = np.zeros_like(x)
    out[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        out[t] = x[t] + gamma * out[t + 1]
    return out


# -----------------------------
# CSV -> Trajectories builder
# -----------------------------

import numpy as np
import pandas as pd

def load_ipd_csv_as_trajectories(csv_path: str, history_k: int = 1):
    """
    Load IPD trajectories from CSV.
    
    Args:
        csv_path: Path to CSV file
        history_k: Number of history decision columns to include in state (default: 1)
                   When history_k=2, includes: my.decision1, other.decision1, 
                   my.decision2, other.decision2
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]

    period_col = "period"
    action_col = "my.decision"

    # Build state columns: base + decision history (NO payoffs)
    state_cols = [
        "risk", "error", "delta", "infin", "contin", "r1", "r2", "r", "s", "t", "p"
    ]
    # Add decision history columns
    for k in range(1, history_k + 1):
        state_cols.extend([f"my.decision{k}", f"other.decision{k}"])


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



def main(args):
    device = args.device
    print(f"Using device: {device}")
    
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # Load CSV -> trajectories
    trajectories, max_ep_len = load_ipd_csv_as_trajectories(
        csv_path=args.csv_path,
        history_k=args.history_k,
    )

    returns = np.array([tr["rewards"].sum() for tr in trajectories], dtype=np.float32)
    traj_lens = np.array([tr["lens"] for tr in trajectories], dtype=np.int32)
    num_timesteps = int(traj_lens.sum())

    total_trajs = len(trajectories)
    state_dim_full = trajectories[0]["observations"].shape[1]

    print("=" * 50)
    print(f"IPD-CSV dataset: {total_trajs} trajectories, {num_timesteps} timesteps")
    print(f"State dim = {state_dim_full}, Action dim = 1")
    print(f"Avg return (all): {returns.mean():.3f} ± {returns.std():.3f} | "
          f"Max: {returns.max():.3f} | Min: {returns.min():.3f}")
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

    def train_single_fold(fold_idx: int, train_inds: np.ndarray, val_inds: np.ndarray) -> tuple[float | None, object]:
        
        print(f"\n--- Fold {fold_idx + 1}/{num_folds} ---")
        print(f"Train trajectories: {len(train_inds)} | Val trajectories: {len(val_inds)}")

        # Define structure state indices
        # State structure: ["risk", "error", "delta", "infin", "contin", "r1", "r2", "r", "s", "t", "p", 
        #                    "my.decision1", "other.decision1", "my.decision2", "other.decision2", ...]
        risk_idx = 0
        error_idx = 1
        delta_idx = 2
        infinity_idx = 3  # "infin"
        continuous_idx = 4  # "contin"

        train_returns = returns[train_inds]
        train_traj_lens = traj_lens[train_inds]
        train_states = np.concatenate([trajectories[int(i)]["observations"] for i in train_inds], axis=0)
        state_mean = train_states.mean(axis=0)
        state_std = train_states.std(axis=0) + 1e-6
        
        # Create validation trajectories list
        val_trajs = [trajectories[int(i)] for i in val_inds]

        p_sample = train_traj_lens / train_traj_lens.sum()

        state_dim = train_states.shape[1]
        act_dim = 1
        scale = max(1.0, np.percentile(train_returns, 95))

        K = args.K
        max_len = K
        num_traj_keep = train_inds.shape[0]

        def make_padded_sample(traj, start_idx: int):
            s_i = traj["observations"][start_idx:start_idx + max_len]
            a_i = traj["actions"][start_idx:start_idx + max_len]
            r_i = traj["rewards"][start_idx:start_idx + max_len]
            d_i = traj["terminals"][start_idx:start_idx + max_len]

            ts = np.arange(start_idx, start_idx + s_i.shape[0], dtype=np.int64)
            ts[ts >= max_ep_len] = max_ep_len - 1

            rtg_i = discount_cumsum(traj["rewards"][start_idx:], gamma=1.0)[: s_i.shape[0] + 1]
            if rtg_i.shape[0] <= s_i.shape[0]:
                rtg_i = np.concatenate([rtg_i, np.zeros((1,), dtype=np.float32)], axis=0)

            tlen = s_i.shape[0]

            s_pad = np.zeros((max_len - tlen, state_dim), dtype=np.float32)
            a_pad = np.ones((max_len - tlen, act_dim), dtype=np.float32) * -10.0
            r_pad = np.zeros((max_len - tlen, 1), dtype=np.float32)
            d_pad = np.ones((max_len - tlen,), dtype=np.int64) * 2
            rtg_pad = np.zeros((max_len - tlen, 1), dtype=np.float32)
            ts_pad = np.zeros((max_len - tlen,), dtype=np.int64)

            s_i = np.concatenate([s_pad, s_i], axis=0).astype(np.float32)
            s_i = (s_i - state_mean) / state_std
            a_i = np.concatenate([a_pad, a_i], axis=0).astype(np.float32)
            r_i = np.concatenate([r_pad, r_i.reshape(-1, 1)], axis=0).astype(np.float32)
            d_i = np.concatenate([d_pad, d_i], axis=0).astype(np.int64)
            rtg_i = np.concatenate([rtg_pad, rtg_i.reshape(-1, 1)], axis=0).astype(np.float32) / float(scale)
            ts = np.concatenate([ts_pad, ts], axis=0).astype(np.int64)
            mask = np.concatenate(
                [np.zeros((max_len - tlen,)), np.ones((tlen,))], axis=0
            ).astype(np.float32)

            return s_i, a_i, r_i, d_i, rtg_i, ts, mask

        def get_batch(batch_size=args.batch_size, max_len=max_len):
            batch_inds = np.random.choice(
                np.arange(num_traj_keep),
                size=batch_size,
                replace=True,
                p=p_sample,
            )

            s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
            for i in range(batch_size):
                traj = trajectories[int(train_inds[batch_inds[i]])]
                si = random.randint(0, traj["rewards"].shape[0] - 1)
                s_i, a_i, r_i, d_i, rtg_i, ts, m = make_padded_sample(traj, si)

                s.append(s_i[None])
                a.append(a_i[None])
                r.append(r_i[None])
                d.append(d_i[None])
                rtg.append(rtg_i[None])
                timesteps.append(ts[None])
                mask.append(m[None])

            s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
            a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
            r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
            d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
            rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
            timesteps_t = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
            mask_t = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

            return s, a, r, d, rtg, timesteps_t, mask_t

        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=args.embed_dim,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_inner=4 * args.embed_dim,
            activation_function=args.activation_function,
            n_positions=1024,
            resid_pdrop=args.dropout,
            attn_pdrop=args.dropout,
            action_tanh=False,  # Output raw logits for BCEWithLogitsLoss
        )
        model = model.to(device=device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps + 1) / args.warmup_steps, 1),
        )

        # Use BCEWithLogitsLoss for binary classification
        # The loss function will be called with: loss_fn(None, action_preds, None, None, action_target, None)
        # where action_preds are raw logits and action_target are binary labels (0 or 1)
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        
        def loss_fn(s_hat, a_hat, r_hat, s, a, r):
            # a_hat: raw logits of shape (N, act_dim) where N is number of valid timesteps
            # a: binary targets of shape (N, act_dim) with values 0. or 1.
            # Compute BCE loss element-wise
            loss_raw = bce_loss(a_hat, a)  # (N, act_dim)
            # Average over action dimension and batch
            return loss_raw.mean()
        
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=args.batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=loss_fn,
            eval_fns=[],
        )

        def compute_val_accuracy():
            if val_inds.size == 0:
                return None

            was_training = model.training
            model.eval()

            total_correct = 0
            total_count = 0

            with torch.no_grad():
                for idx in val_inds:
                    traj = trajectories[int(idx)]
                    T = traj["rewards"].shape[0]

                    for si in range(T):
                        s_i, a_i, r_i, d_i, rtg_i, ts, m = make_padded_sample(traj, si)

                        s_batch = torch.from_numpy(s_i[None]).to(dtype=torch.float32, device=device)
                        a_batch = torch.from_numpy(a_i[None]).to(dtype=torch.float32, device=device)
                        r_batch = torch.from_numpy(r_i[None]).to(dtype=torch.float32, device=device)
                        rtg_batch = torch.from_numpy(rtg_i[None]).to(dtype=torch.float32, device=device)
                        ts_batch = torch.from_numpy(ts[None]).to(dtype=torch.long, device=device)
                        mask_batch = torch.from_numpy(m[None]).to(dtype=torch.float32, device=device)

                        _, action_preds, _ = model.forward(
                            s_batch,
                            a_batch,
                            r_batch,
                            rtg_batch[:, :-1],
                            ts_batch,
                            attention_mask=mask_batch,
                        )
                        act_dim_eval = action_preds.shape[2]
                        action_preds = action_preds.reshape(-1, act_dim_eval)
                        action_target = a_batch.reshape(-1, act_dim_eval)
                        mask_flat = mask_batch.reshape(-1).bool()
                        action_preds = action_preds[mask_flat]
                        action_target = action_target[mask_flat]

                        if action_preds.shape[0] == 0:
                            continue

                        # Apply sigmoid to logits to get probabilities, then threshold at 0.5
                        action_probs = torch.sigmoid(action_preds)
                        action_preds_binary = (action_probs >= 0.5).float()
                        
                        # Calculate number of correct predictions
                        correct = (action_preds_binary == action_target).sum().item()
                        total_correct += correct
                        total_count += action_preds.shape[0]

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
            group_name = "ipd-csv"
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
            logs = trainer.train_iteration(num_steps=args.num_steps_per_iter, iter_num=itn, print_logs=True)
            if logs is None:
                logs = {}
            val_accuracy = compute_val_accuracy()
            if val_accuracy is not None:
                logs["validation/accuracy"] = val_accuracy
                last_val_accuracy = val_accuracy
                print(f"Validation Accuracy: {val_accuracy:.6f}")
            if logs and args.log_to_wandb:
                wandb.log(logs)

        print("\n" + "=" * 70)
        print(f"Evaluating IPD metrics for Fold {fold_idx + 1}/{num_folds}")
        print("=" * 70)
        metrics = evaluate_ipd_metrics(
            model=model,
            trajectories=val_trajs,
            K=args.K,
            max_ep_len=max_ep_len,
            device=device,
            state_mean=state_mean,
            state_std=state_std,
            structure_state_idx=(risk_idx, error_idx, delta_idx, infinity_idx, continuous_idx),
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
        print(f"Average validation Accuracy over {len(fold_val_accuracies)} fold(s): {avg_val_accuracy:.6f}")
        print("=" * 50)
    
    # Evaluate on test set (10%)
    if len(test_inds) > 0 and len(trained_models) > 0:
        print("\n" + "=" * 70)
        print("Evaluating on Test Set (10% held-out)")
        print("=" * 70)
        
        # Use the model from the last fold for test evaluation
        test_model = trained_models[-1]
        test_trajs = [trajectories[int(i)] for i in test_inds]
        
        # Calculate normalization stats from all CV data
        all_cv_states = np.concatenate([trajectories[int(i)]["observations"] for i in cv_inds], axis=0)
        test_state_mean = all_cv_states.mean(axis=0)
        test_state_std = all_cv_states.std(axis=0) + 1e-6
        
        # Define structure state indices
        risk_idx = 0
        error_idx = 1
        delta_idx = 2
        infinity_idx = 3
        continuous_idx = 4
        
        test_metrics = evaluate_ipd_metrics(
            model=test_model,
            trajectories=test_trajs,
            K=args.K,
            max_ep_len=max_ep_len,
            device=device,
            state_mean=test_state_mean,
            state_std=test_state_std,
            structure_state_idx=(risk_idx, error_idx, delta_idx, infinity_idx, continuous_idx),
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

        if args.log_to_wandb:
            wandb_api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY")
            if wandb_api_key:
                os.environ["WANDB_API_KEY"] = wandb_api_key
                wandb.login(key=wandb_api_key, relogin=True)
                group_name = "ipd-dt1"
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
    parser.add_argument("--history_k", type=int, default=3, help="Number of history decision columns to include in state (default: 1). When history_k=2, includes my.decision1-2 and other.decision1-2")
    parser.add_argument("--K", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
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
