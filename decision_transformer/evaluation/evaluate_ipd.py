# evaluate_ipd.py
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

def _sigmoid_safe(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.sigmoid(x), 1e-6, 1 - 1e-6)

def _binary_ll(y_true: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return (y_true * torch.log(p) + (1 - y_true) * torch.log(1 - p)).sum()

# Base MO features (without history)
MODUS_OPERANDI_BASE_COLS = [
    "r1", "r2", "risk", "error", "delta", "r1*delta", "r2*delta", "infin", "contin", "delta*infin"
]

def _build_feature_names(history_k: int) -> List[str]:
    """
    Build MO feature names dynamically based on history_k.
    
    Args:
        history_k: Number of history decision columns to include
        
    Returns:
        List of feature names matching the order used in load_ipd_csv_as_trajectories
    """
    feature_names = MODUS_OPERANDI_BASE_COLS.copy()
    for k in range(1, history_k + 1):
        feature_names.append(f"my.decision{k}")
        feature_names.append(f"other.decision{k}")
    return feature_names

def evaluate_ipd_metrics_and_betas(
    model,
    trajectories: List[Dict],
    *,
    K: int,
    max_ep_len: int,
    device: torch.device,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    mo_mean: np.ndarray = None,
    mo_std: np.ndarray = None,
    history_k: Optional[int] = None,  # Number of history decision columns
    structure_state_idx: Tuple[int, int, int, int, int] = (0,1,2,3,4),  # indices in state for risk,error,delta,infinity,continuous
    print_report: bool = True,
) -> Dict[str, float]:

    model.eval()
    state_mean_t = torch.as_tensor(state_mean, device=device, dtype=torch.float32)
    state_std_t  = torch.as_tensor(state_std,  device=device, dtype=torch.float32)
    mo_mean_t = torch.as_tensor(mo_mean, device=device, dtype=torch.float32) if mo_mean is not None else None
    mo_std_t  = torch.as_tensor(mo_std,  device=device, dtype=torch.float32) if mo_std is not None else None

    # Determine history_k from MO dimension if not provided
    if history_k is None:
        if len(trajectories) > 0 and "modus_operandi" in trajectories[0]:
            mo_dim = trajectories[0]["modus_operandi"].shape[1]
            # MO dim = 10 base features + 2 * history_k (my.decision{k} + other.decision{k})
            # Solve: mo_dim = 10 + 2 * history_k
            # history_k = (mo_dim - 10) / 2
            if mo_dim >= 10 and (mo_dim - 10) % 2 == 0:
                history_k = (mo_dim - 10) // 2
            else:
                # Fallback: assume history_k=1 if dimension doesn't match expected pattern
                history_k = 1
        else:
            history_k = 1
    
    # Build feature names dynamically based on history_k
    FEATURE_NAMES = _build_feature_names(history_k)
    
    # Verify that feature names match MO dimension
    expected_mo_dim = len(FEATURE_NAMES)
    if len(trajectories) > 0 and "modus_operandi" in trajectories[0]:
        actual_mo_dim = trajectories[0]["modus_operandi"].shape[1]
        if actual_mo_dim != expected_mo_dim:
            print(f"Warning: MO dimension mismatch. Expected {expected_mo_dim} features (history_k={history_k}), "
                  f"but trajectory has {actual_mo_dim} features. Beta coefficient extraction may be incorrect.")

    y_t1,  p_t1  = [], []
    y_tgt, p_tgt = [], []

    per_time_truth,  per_time_pred  = {}, {}
    per_struct_truth, per_struct_pred = {}, {}

    # Detect model type to determine window size
    import inspect
    forward_sig = inspect.signature(model.forward)
    forward_params = list(forward_sig.parameters.keys())
    is_gpt2 = 'returns_to_go' not in forward_params
    # For GPT2BCModel, use model's max_length instead of K
    eval_window_size = model.max_length if (is_gpt2 and hasattr(model, 'max_length')) else K

    with torch.no_grad():
        for traj in trajectories:
            S = traj["observations"].astype(np.float32)    # (T, state_dim)
            A = traj["actions"].astype(np.float32)         # (T, 1)  {0,1}
            R = traj["rewards"].astype(np.float32)         # (T,)
            MO = traj["modus_operandi"].astype(np.float32) # (T, mo_dim) — raw MO *features*

            T = S.shape[0]
            state_dim = S.shape[1]
            mo_dim    = MO.shape[1]

            for si in range(T):
                tlen = min(eval_window_size, T - si)

                s_i  = S[si:si+tlen]
                a_i  = A[si:si+tlen]
                r_i  = R[si:si+tlen]
                mo_i = MO[si:si+tlen]

                # pad to length eval_window_size (K for DT, model.max_length for GPT2BCModel)
                s_pad  = np.zeros((eval_window_size - tlen, state_dim), dtype=np.float32)
                a_pad  = np.ones((eval_window_size - tlen, 1), dtype=np.float32) * -10.0
                r_pad  = np.zeros((eval_window_size - tlen, 1), dtype=np.float32)
                mo_pad = np.zeros((eval_window_size - tlen, mo_dim), dtype=np.float32)
                ts_pad = np.zeros((eval_window_size - tlen,), dtype=np.int64)

                s_win  = np.concatenate([s_pad, s_i], axis=0)
                a_win  = np.concatenate([a_pad, a_i], axis=0)
                r_win  = np.concatenate([r_pad, r_i.reshape(-1,1)], axis=0)
                mo_win = np.concatenate([mo_pad, mo_i], axis=0)

                # simple RTG (γ=1) - only needed for DecisionTransformer
                rtg = np.cumsum(r_i[::-1])[::-1].astype(np.float32)
                if rtg.shape[0] < tlen + 1:
                    rtg = np.concatenate([rtg, np.zeros((1,), dtype=np.float32)], 0)
                rtg = np.concatenate([np.zeros((eval_window_size - tlen, 1), dtype=np.float32),
                                      rtg.reshape(-1,1)], 0)

                ts = np.arange(si, si + tlen, dtype=np.int64)
                ts = np.minimum(ts, max_ep_len - 1)
                ts = np.concatenate([ts_pad, ts], 0)

                mask = np.concatenate([np.zeros((eval_window_size - tlen,)), np.ones((tlen,))], 0).astype(np.float32)

                # to tensors
                s_t   = torch.as_tensor(s_win, device=device, dtype=torch.float32)
                a_t   = torch.as_tensor(a_win, device=device, dtype=torch.float32)
                r_t   = torch.as_tensor(r_win, device=device, dtype=torch.float32)
                mo_t  = torch.as_tensor(mo_win, device=device, dtype=torch.float32)
                rtg_t = torch.as_tensor(rtg, device=device, dtype=torch.float32)
                ts_t  = torch.as_tensor(ts,  device=device, dtype=torch.long)
                m_t   = torch.as_tensor(mask,device=device, dtype=torch.float32)

                # normalize states as in training
                s_t = (s_t - state_mean_t) / state_std_t
                
                # normalize MO features as in training (if normalization params provided)
                if mo_mean_t is not None and mo_std_t is not None:
                    mo_t = (mo_t - mo_mean_t) / mo_std_t

                # Call model.forward with appropriate arguments based on model type
                if 'mo' in forward_params:
                    # DecisionTransformer with MO support
                    _, a_pred, _ = model.forward(
                        s_t.unsqueeze(0), a_t.unsqueeze(0), None,
                        rtg_t.unsqueeze(0)[:, :-1], ts_t.unsqueeze(0),
                        attention_mask=m_t.unsqueeze(0), mo=mo_t.unsqueeze(0)
                    )
                elif 'returns_to_go' not in forward_params:
                    # GPT2BCModel: forward(states, actions, rewards, timesteps=None, attention_mask=None, target_return=None)
                    # Note: timesteps is positional with default, so we need to pass it explicitly as keyword arg
                    _, a_pred, _ = model.forward(
                        s_t.unsqueeze(0), a_t.unsqueeze(0), None,
                        timesteps=ts_t.unsqueeze(0), attention_mask=m_t.unsqueeze(0)
                    )
                else:
                    # DecisionTransformer without MO
                    _, a_pred, _ = model.forward(
                        s_t.unsqueeze(0), a_t.unsqueeze(0), None,
                        rtg_t.unsqueeze(0)[:, :-1], ts_t.unsqueeze(0),
                        attention_mask=m_t.unsqueeze(0)
                    )

                # Handle different output shapes: DT returns (B, T, act_dim), GPT2 returns (B, 1, act_dim)
                if a_pred.shape[1] == 1:
                    # GPT2BCModel: (B, 1, act_dim) - only predicts next action after the sequence
                    # For evaluation, we evaluate at the last valid timestep (the one GPT2 is predicting for)
                    a_pred = a_pred.squeeze(1).squeeze(0)  # (act_dim,) - remove batch and seq dims
                    # Get the last valid action from the sequence (this is what GPT2 is predicting)
                    mask_bool = m_t.bool()  # (K,)
                    if mask_bool.any():
                        last_valid_idx = mask_bool.nonzero()[-1].item()
                        y_true = a_t[last_valid_idx]  # (act_dim,)
                        # Get the absolute time index for the last valid position
                        mask_bool_np = mask_bool.cpu().numpy()
                        abs_times = ts[mask_bool_np] - si + (si + 1)
                        t_abs = int(abs_times[-1]) if len(abs_times) > 0 else si + 1
                        struct_vals = s_win[mask_bool_np][:, list(structure_state_idx)]
                        key = tuple(struct_vals[-1].tolist()) if len(struct_vals) > 0 else tuple([0]*len(structure_state_idx))
                        
                        # Single prediction for GPT2
                        yt = float(y_true.item())
                        pp = float(_sigmoid_safe(a_pred.unsqueeze(0)).item())
                        
                        if t_abs == 1:
                            y_t1.append(yt);  p_t1.append(pp)
                        else:
                            y_tgt.append(yt); p_tgt.append(pp)

                        per_time_truth.setdefault(t_abs, []).append(yt)
                        per_time_pred.setdefault(t_abs, []).append(pp)
                        per_struct_truth.setdefault(key, []).append(yt)
                        per_struct_pred.setdefault(key, []).append(pp)
                    else:
                        continue  # Skip if no valid positions
                else:
                    # DecisionTransformer: (B, T, act_dim)
                    a_pred = a_pred.reshape(eval_window_size, -1)[m_t.bool()]           # (tlen, 1)
                    y_true = a_t.reshape(eval_window_size, -1)[m_t.bool()]              # (tlen, 1)

                    p = _sigmoid_safe(a_pred.squeeze(-1))                # (tlen,)
                    y = y_true.squeeze(-1)                               # (tlen,)

                    # absolute time index (1-based)
                    # Convert mask to numpy for indexing numpy array ts
                    mask_bool_np = m_t.bool().cpu().numpy()
                    abs_times = ts[mask_bool_np] - si + (si + 1)
                    struct_vals = s_win[mask_bool_np][:, list(structure_state_idx)]

                    for k in range(p.shape[0]):
                        yt, pp = float(y[k].item()), float(p[k].item())
                        t_abs  = int(abs_times[k])
                        key    = tuple(struct_vals[k].tolist())

                        if t_abs == 1:
                            y_t1.append(yt);  p_t1.append(pp)
                        else:
                            y_tgt.append(yt); p_tgt.append(pp)

                        per_time_truth.setdefault(t_abs, []).append(yt)
                        per_time_pred.setdefault(t_abs, []).append(pp)
                        per_struct_truth.setdefault(key, []).append(yt)
                        per_struct_pred.setdefault(key, []).append(pp)

    def _acc_ll(y_list, p_list):
        if not y_list:
            return 0.0, 0.0
        y = torch.tensor(y_list, dtype=torch.float32)
        p = torch.tensor(p_list, dtype=torch.float32)
        acc = ((p >= 0.5).float() == y).float().mean().item()
        ll  = _binary_ll(y, p).item()
        return acc, ll

    acc_t1,  ll_t1  = _acc_ll(y_t1,  p_t1)
    acc_tgt, ll_tgt = _acc_ll(y_tgt, p_tgt)

    def _agg(d_truth: Dict[int, List[float]], d_pred: Dict[int, List[float]]):
        keys = sorted(set(d_truth.keys()) & set(d_pred.keys()))
        if not keys:
            return 0.0, 0.0
        truth = np.array([np.mean(d_truth[k]) for k in keys], dtype=np.float64)
        pred  = np.array([np.mean(d_pred[k]) for k in keys],  dtype=np.float64)
        corr = 0.0 if (truth.std()<1e-12 or pred.std()<1e-12) else float(np.corrcoef(truth, pred)[0,1])
        rmse = float(np.sqrt(np.mean((truth - pred)**2)))
        return corr, rmse

    cor_time, rmse_time = _agg(per_time_truth, per_time_pred)
    cor_avg,  rmse_avg  = _agg(per_struct_truth, per_struct_pred)

    # ---- NEW: pull MO β coefficients from the model (only for DecisionTransformer with MO) ----
    beta_table = {}
    if hasattr(model, 'get_mo_beta_coefficients'):
        try:
            beta_w, beta_b = model.get_mo_beta_coefficients()  # (mo_dim,), scalar
            beta_w = beta_w.detach().cpu().numpy().astype(float)
            beta_b = float(beta_b)
            
            # Ensure beta_w has the same length as FEATURE_NAMES
            if len(beta_w) != len(FEATURE_NAMES):
                print(f"Warning: Beta coefficients dimension ({len(beta_w)}) doesn't match feature names ({len(FEATURE_NAMES)}). "
                      f"Truncating or padding to match.")
                if len(beta_w) > len(FEATURE_NAMES):
                    beta_w = beta_w[:len(FEATURE_NAMES)]
                else:
                    # Pad with zeros if beta_w is shorter (shouldn't happen normally)
                    beta_w = np.pad(beta_w, (0, len(FEATURE_NAMES) - len(beta_w)), mode='constant', constant_values=0.0)
            
            beta_table = {name: w for name, w in zip(FEATURE_NAMES, beta_w)}
            beta_table["bias"] = beta_b
        except Exception as e:
            print(f"Warning: Could not extract beta coefficients: {e}")

    report = {
        "Acc.t=1": acc_t1, "Acc.t>1": acc_tgt,
        "LL.t=1": ll_t1,   "LL.t>1": ll_tgt,
        "Cor-Time": cor_time, "RMSE-Time": rmse_time,
        "Cor-Avg":  cor_avg,  "RMSE-Avg":  rmse_avg,
        **{f"beta/{k}": v for k, v in beta_table.items()},
    }

    if print_report:
        metrics_str = ", ".join([
            f"Acc.t=1 {acc_t1:.3f}", f"Acc.t>1 {acc_tgt:.3f}",
            f"LL.t=1 {ll_t1:.0f}",   f"LL.t>1 {ll_tgt:.0f}",
            f"Cor-Time {cor_time:.3f}", f"RMSE-Time {rmse_time:.3f}",
            f"Cor-Avg {cor_avg:.3f}",   f"RMSE-Avg {rmse_avg:.3f}",
        ])
        print("IPD Eval →", metrics_str)
        if beta_table:
            print("MO β coefficients:")
            for name in FEATURE_NAMES:
                if name in beta_table:
                    print(f"  β_{name:9s} = {beta_table[name]: .6f}")
            if "bias" in beta_table:
                print(f"  β_bias     = {beta_table['bias']: .6f}")

    return report
