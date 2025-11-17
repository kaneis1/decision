# evaluate_mo_formula.py
# Evaluate the original MO formula with logistic sigmoid
# Compare with Decision Transformer model

import numpy as np
import torch
from typing import Dict, List, Tuple

def _sigmoid_safe(x: np.ndarray) -> np.ndarray:
    """Safe sigmoid: 1 / (1 + exp(-x)), clamped to avoid numerical issues."""
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))), 1e-6, 1 - 1e-6)

def _binary_ll(y_true: np.ndarray, p: np.ndarray) -> float:
    """Binary log-likelihood."""
    return float(np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

def evaluate_mo_formula(
    trajectories: List[Dict],
    K: int,
    max_ep_len: int,
    structure_state_idx: Tuple[int, int, int, int, int] = (0, 1, 2, 3, 4),
    print_report: bool = True,
) -> Dict[str, float]:
    """
    Evaluate the original MO formula with logistic sigmoid.
    
    Formula for C_t = 1:
        r1 + r2 + risk + error + delta + r1*delta + r2*delta + infin + contin
    
    Formula for C_t > 1:
        r1 + r2 + risk + error + delta + r1*delta + r2*delta + infin + contin 
        + delta*infin + my.decision_t-1 + other.decision_t-1 + error*other.decision_t-1 + t
    
    Then apply logistic sigmoid: σ(w^T X) = 1 / (1 + exp(-w^T X))
    With equal weights (all 1), this simplifies to: σ(sum(X)) = 1 / (1 + exp(-sum(X)))
    
    Args:
        trajectories: List of trajectory dicts with keys:
            - observations: (T, state_dim) - states
            - actions: (T, 1) - ground truth actions {0,1}
            - rewards: (T,) - rewards
            - modus_operandi: (T, mo_dim) - MO features
        K: Context length (window size)
        max_ep_len: Maximum episode length
        structure_state_idx: Tuple of (risk_idx, error_idx, delta_idx, infinity_idx, continuous_idx)
        print_report: Whether to print evaluation report
    
    Returns:
        Dictionary of metrics matching DT evaluation format
    """
    # MO feature indices according to MODUS_OPERANDI_COLS_DEFAULT:
    # ["r1","r2","risk","error","delta","r1*delta","r2*delta","infin","contin","delta*infin","my.decision1","other.decision1"]
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
    
    # Structure state indices (from observations)
    risk_idx, error_idx, delta_idx, infinity_idx, continuous_idx = structure_state_idx
    
    y_t1, p_t1 = [], []
    y_tgt, p_tgt = [], []
    per_time_truth, per_time_pred = {}, {}
    per_struct_truth, per_struct_pred = {}, {}
    
    for traj in trajectories:
        S = traj["observations"].astype(np.float32)    # (T, state_dim)
        A = traj["actions"].astype(np.float32)         # (T, 1)  {0,1}
        R = traj["rewards"].astype(np.float32)         # (T,)
        MO = traj["modus_operandi"].astype(np.float32) # (T, mo_dim)
        
        T = S.shape[0]
        
        # Get period numbers - check if period is in observations
        # STATE_COLS_DEFAULT = ["risk", "error", "delta", "infin", "contin", "r1", "r2", "r", "s", "t", "p", ...]
        # Period "t" is at index 9 (0-based)
        # Period values from CSV are typically already 1-based
        if S.shape[1] > 9:
            periods = S[:, 9].astype(np.float32)  # Use period directly from observations
            # Ensure periods are valid (>= 1, in case they're 0-based)
            periods = np.maximum(periods, np.arange(1, T + 1, dtype=np.float32))
        else:
            periods = np.arange(1, T + 1, dtype=np.float32)  # 1-based periods (fallback)
        
        for si in range(T):
            tlen = min(K, T - si)
            
            # Extract window
            mo_window = MO[si:si+tlen]  # (tlen, mo_dim)
            a_window = A[si:si+tlen]    # (tlen, 1)
            s_window = S[si:si+tlen]    # (tlen, state_dim)
            period_window = periods[si:si+tlen]  # (tlen,)
            
            # Compute MO formula for each timestep in window
            for t_idx in range(tlen):
                t_abs = si + t_idx + 1  # 1-based absolute time
                period = period_window[t_idx]
                
                # Extract MO features
                r1 = mo_window[t_idx, MO_IDX_R1]
                r2 = mo_window[t_idx, MO_IDX_R2]
                risk = mo_window[t_idx, MO_IDX_RISK]
                error = mo_window[t_idx, MO_IDX_ERROR]
                delta = mo_window[t_idx, MO_IDX_DELTA]
                r1_delta = mo_window[t_idx, MO_IDX_R1_DELTA]
                r2_delta = mo_window[t_idx, MO_IDX_R2_DELTA]
                infin = mo_window[t_idx, MO_IDX_INFIN]
                contin = mo_window[t_idx, MO_IDX_CONTIN]
                
                # Base features (for both t=1 and t>1)
                mo_sum = r1 + r2 + risk + error + delta + r1_delta + r2_delta + infin + contin
                
                # Additional features for t > 1
                if period > 1:
                    delta_infin = mo_window[t_idx, MO_IDX_DELTA_INFIN]
                    my_decision_t1 = mo_window[t_idx, MO_IDX_MY_DECISION1]
                    other_decision_t1 = mo_window[t_idx, MO_IDX_OTHER_DECISION1]
                    
                    # Handle missing values (filled with 2 in data loading for first period)
                    # For t>1, these should be valid (0 or 1), but check anyway
                    if my_decision_t1 == 2:
                        my_decision_t1 = 0.0  # Default to 0 if missing
                    if other_decision_t1 == 2:
                        other_decision_t1 = 0.0  # Default to 0 if missing
                    
                    # Get error from MO (already extracted above)
                    error_other_t1 = error * other_decision_t1
                    
                    # Add additional terms
                    mo_sum += delta_infin + my_decision_t1 + other_decision_t1 + error_other_t1 + period
                
                # Apply logistic sigmoid: σ(sum) = 1 / (1 + exp(-sum))
                p_action = _sigmoid_safe(np.array([mo_sum]))[0]
                
                # Ground truth
                y_true = float(a_window[t_idx, 0])
                
                # Store predictions
                if period == 1:
                    y_t1.append(y_true)
                    p_t1.append(p_action)
                else:
                    y_tgt.append(y_true)
                    p_tgt.append(p_action)
                
                per_time_truth.setdefault(t_abs, []).append(y_true)
                per_time_pred.setdefault(t_abs, []).append(p_action)
                
                # Structure state key (for grouping by game parameters)
                struct_key = tuple(s_window[t_idx, [risk_idx, error_idx, delta_idx, infinity_idx, continuous_idx]].tolist())
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
        metrics_str = ", ".join([
            f"Acc.t=1 {acc_t1:.3f}", f"Acc.t>1 {acc_tgt:.3f}",
            f"LL.t=1 {ll_t1:.0f}",   f"LL.t>1 {ll_tgt:.0f}",
            f"Cor-Time {cor_time:.3f}", f"RMSE-Time {rmse_time:.3f}",
            f"Cor-Avg {cor_avg:.3f}",   f"RMSE-Avg {rmse_avg:.3f}",
        ])
        print("MO Formula Eval →", metrics_str)
    
    return report

