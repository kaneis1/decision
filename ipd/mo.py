import argparse
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


def discount_cumsum(x, gamma: float):
    """Same helper as experiment.py (gamma=1. for RTG)."""
    out = np.zeros_like(x)
    out[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        out[t] = x[t] + gamma * out[t + 1]
    return out


def _sigmoid_safe(x: np.ndarray) -> np.ndarray:
    """Safe sigmoid: 1 / (1 + exp(-x)), clamped to avoid numerical issues."""
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))), 1e-6, 1 - 1e-6)


def _binary_ll(y_true: np.ndarray, p: np.ndarray) -> float:
    """Binary log-likelihood."""
    return float(np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


MODUS_OPERANDI_BASE_COLS = [
    "r1", "r2", "risk", "error", "delta", "r1*delta", "r2*delta", "infin", "contin", "delta*infin", "my.decision1", "other.decision1"
]


def load_ipd_csv_as_trajectories(csv_path: str, history_k: int = 1):
    """
    Load IPD trajectories from CSV.
    
    Args:
        csv_path: Path to CSV file
        history_k: Number of history decision columns to include in MO (default: 1)
                   When history_k=3, includes: my.decision1, other.decision1, 
                   my.decision2, other.decision2, my.decision3, other.decision3
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]

    period_col = "period"
    action_col = "my.decision"

    state_cols = [
        "risk", "error", "delta", "infin", "contin", "r1", "r2", "r", "s", "t", "p",
        "my.decision1", "other.decision1"
    ]
    
    # Build MO columns dynamically based on history_k
    modus_operandi_cols = ["r1", "r2", "risk", "error", "delta", "r1*delta", "r2*delta", "infin", "contin", "delta*infin"]
    for k in range(1, history_k + 1):
        modus_operandi_cols.extend([f"my.decision{k}", f"other.decision{k}"])

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

        trajectories.append(dict(
            observations=s,
            actions=a,
            rewards=r,
            modus_operandi=mo,
            terminals=terminals,
            lens=int(T)
        ))
        max_ep_len = max(max_ep_len, T)

    return trajectories, max_ep_len


def compute_mo_formula(mo_features: np.ndarray, period: float, mo_idx_map: Dict[str, int]) -> float:
    """
    Compute MO formula according to the two equations:
    
    For Ct = 1:
        r1 + r2 + risk + error + delta + r1*delta + r2*delta + infinity + continuous
    
    For Ct > 1:
        All above + delta*infinity + my.decision_{t-1} + other.decision_{t-1} + error*other.decision_{t-1} + t
    
    Args:
        mo_features: Array of MO feature values (1D array)
        period: Current period (1-based)
        mo_idx_map: Dictionary mapping feature names to indices in mo_features
    
    Returns:
        Sum of features (before sigmoid)
    """
    # Base features for both t=1 and t>1
    mo_sum = (
        mo_features[mo_idx_map["r1"]] +
        mo_features[mo_idx_map["r2"]] +
        mo_features[mo_idx_map["risk"]] +
        mo_features[mo_idx_map["error"]] +
        mo_features[mo_idx_map["delta"]] +
        mo_features[mo_idx_map["r1*delta"]] +
        mo_features[mo_idx_map["r2*delta"]] +
        mo_features[mo_idx_map["infin"]] +
        mo_features[mo_idx_map["contin"]]
    )
    
    # Additional features for t > 1
    if period > 1:
        my_decision_t1 = mo_features[mo_idx_map["my.decision1"]]
        other_decision_t1 = mo_features[mo_idx_map["other.decision1"]]
        error = mo_features[mo_idx_map["error"]]
        
        # Handle missing values (filled with 2 in data loading for first period)
        if my_decision_t1 == 2:
            my_decision_t1 = 0.0
        if other_decision_t1 == 2:
            other_decision_t1 = 0.0
        
        mo_sum += (
            mo_features[mo_idx_map["delta*infin"]] +
            my_decision_t1 +
            other_decision_t1 +
            error * other_decision_t1 +
            period
        )
    
    return mo_sum


def evaluate_mo_formula_metrics(
    trajectories: List[Dict],
    structure_state_idx: Tuple[int, int, int, int, int] = (0, 1, 2, 3, 4),
    print_report: bool = True,
) -> Dict[str, float]:
    """
    Evaluate MO formula with sigmoid and calculate metrics.
    
    Args:
        trajectories: List of trajectory dicts
        structure_state_idx: Tuple of (risk_idx, error_idx, delta_idx, infinity_idx, continuous_idx)
        print_report: Whether to print evaluation report
    
    Returns:
        Dictionary of metrics
    """
    # MO feature indices (assuming history_k=1 for now, can be extended)
    MO_IDX_R1 = 0
    MO_IDX_R2 = 1
    MO_IDX_RISK = 2
    MO_IDX_ERROR = 3
    MO_IDX_DELTA = 4
    MO_IDX_R1_DELTA = 5
    MO_IDX_R2_DELTA = 6
    MO_IDX_INFIN = 7
    MO_IDX_CONTIN = 8
    MO_IDX_DELTA_INFIN = 9
    MO_IDX_MY_DECISION1 = 10
    MO_IDX_OTHER_DECISION1 = 11
    
    mo_idx_map = {
        "r1": MO_IDX_R1,
        "r2": MO_IDX_R2,
        "risk": MO_IDX_RISK,
        "error": MO_IDX_ERROR,
        "delta": MO_IDX_DELTA,
        "r1*delta": MO_IDX_R1_DELTA,
        "r2*delta": MO_IDX_R2_DELTA,
        "infin": MO_IDX_INFIN,
        "contin": MO_IDX_CONTIN,
        "delta*infin": MO_IDX_DELTA_INFIN,
        "my.decision1": MO_IDX_MY_DECISION1,
        "other.decision1": MO_IDX_OTHER_DECISION1,
    }
    
    # Structure state indices (from observations)
    risk_idx, error_idx, delta_idx, infinity_idx, continuous_idx = structure_state_idx
    
    y_t1, p_t1 = [], []
    y_tgt, p_tgt = [], []
    per_time_truth, per_time_pred = {}, {}
    per_struct_truth, per_struct_pred = {}, {}
    
    for traj in trajectories:
        S = traj["observations"].astype(np.float32)    # (T, state_dim)
        A = traj["actions"].astype(np.float32)         # (T, 1)  {0,1}
        MO = traj["modus_operandi"].astype(np.float32) # (T, mo_dim)
        
        T = S.shape[0]
        
        # Get period numbers from observations (period "t" is at index 9)
        if S.shape[1] > 9:
            periods = S[:, 9].astype(np.float32)
            periods = np.maximum(periods, np.arange(1, T + 1, dtype=np.float32))
        else:
            periods = np.arange(1, T + 1, dtype=np.float32)  # 1-based periods (fallback)
        
        for t_idx in range(T):
            period = periods[t_idx]
            mo_features = MO[t_idx]  # (mo_dim,)
            
            # Compute MO formula sum
            mo_sum = compute_mo_formula(mo_features, period, mo_idx_map)
            
            # Apply logistic sigmoid: σ(sum) = 1 / (1 + exp(-sum))
            p_action = float(_sigmoid_safe(np.array([mo_sum]))[0])
            
            # Ground truth
            y_true = float(A[t_idx, 0])
            
            # Store predictions
            if period == 1:
                y_t1.append(y_true)
                p_t1.append(p_action)
            else:
                y_tgt.append(y_true)
                p_tgt.append(p_action)
            
            t_abs = int(period)
            per_time_truth.setdefault(t_abs, []).append(y_true)
            per_time_pred.setdefault(t_abs, []).append(p_action)
            
            # Structure state key (for grouping by game parameters)
            struct_key = tuple(S[t_idx, [risk_idx, error_idx, delta_idx, infinity_idx, continuous_idx]].tolist())
            per_struct_truth.setdefault(struct_key, []).append(y_true)
            per_struct_pred.setdefault(struct_key, []).append(p_action)
    
    def _acc_ll(y_list, p_list):
        if not y_list:
            return 0.0, 0.0
        y = np.array(y_list, dtype=np.float32)
        p = np.array(p_list, dtype=np.float32)
        acc = float(np.mean((p >= 0.5).astype(float) == y))
        ll = _binary_ll(y, p)
        return acc, ll
    
    acc_t1,  ll_t1  = _acc_ll(y_t1,  p_t1)
    acc_tgt, ll_tgt = _acc_ll(y_tgt, p_tgt)
    
    def _agg(d_truth: Dict[int, List[float]], d_pred: Dict[int, List[float]]):
        keys = sorted(set(d_truth.keys()) & set(d_pred.keys()))
        if not keys:
            return 0.0, 0.0
        truth = np.array([np.mean(d_truth[k]) for k in keys], dtype=np.float64)
        pred  = np.array([np.mean(d_pred[k]) for k in keys],  dtype=np.float64)
        corr = 0.0 if (truth.std() < 1e-12 or pred.std() < 1e-12) else float(np.corrcoef(truth, pred)[0, 1])
        rmse = float(np.sqrt(np.mean((truth - pred) ** 2)))
        return corr, rmse
    
    cor_time, rmse_time = _agg(per_time_truth, per_time_pred)
    cor_avg,  rmse_avg  = _agg(per_struct_truth, per_struct_pred)
    
    report = {
        "Acc.t=1": acc_t1, "Acc.t>1": acc_tgt,
        "LL.t=1": ll_t1,   "LL.t>1": ll_tgt,
        "Cor-Time": cor_time, "RMSE-Time": rmse_time,
        "Cor-Avg":  cor_avg,  "RMSE-Avg":  rmse_avg,
    }
    
    if print_report:
        print("\n" + "=" * 70)
        print("MO Formula Evaluation Results")
        print("=" * 70)
        print(f"{'Metric':<15} {'Value':>10}")
        print("-" * 70)
        print(f"{'Acc. t = 1':<15} {acc_t1:>10.3f}")
        print(f"{'Acc. t > 1':<15} {acc_tgt:>10.3f}")
        print(f"{'LL t = 1':<15} {ll_t1:>10.0f}")
        print(f"{'LL t > 1':<15} {ll_tgt:>10.0f}")
        print(f"{'Cor-Time':<15} {cor_time:>10.3f}")
        print(f"{'Cor-Avg.':<15} {cor_avg:>10.3f}")
        print(f"{'RMSE-Time':<15} {rmse_time:>10.3f}")
        print(f"{'RMSE-Avg.':<15} {rmse_avg:>10.3f}")
        print("=" * 70 + "\n")
    
    return report


def main(args):
    device = args.device
    print(f"Using device: {device}")
    
    # Load trajectories
    trajectories, max_ep_len = load_ipd_csv_as_trajectories(
        csv_path=args.csv_path,
        history_k=args.history_k,
    )
    
    print("=" * 50)
    print(f"Loaded {len(trajectories)} trajectories, max_ep_len: {max_ep_len}")
    print("=" * 50)
    
    # Structure state indices: risk, error, delta, infinity, continuous
    # Based on STATE_COLS_DEFAULT = ["risk", "error", "delta", "infin", "contin", ...]
    structure_state_idx = (0, 1, 2, 3, 4)
    
    # Evaluate MO formula
    metrics = evaluate_mo_formula_metrics(
        trajectories=trajectories,
        structure_state_idx=structure_state_idx,
        print_report=True,
    )
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data/all_data.csv")
    parser.add_argument("--history_k", type=int, default=1, help="Number of history decision columns")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
