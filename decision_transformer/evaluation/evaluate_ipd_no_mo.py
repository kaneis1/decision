# evaluate_ipd_no_mo.py
# Evaluation function for standard DecisionTransformer (without MO support)
import numpy as np
import torch
from typing import Dict, List, Tuple

def _sigmoid_safe(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.sigmoid(x), 1e-6, 1 - 1e-6)

def _binary_ll(y_true: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return (y_true * torch.log(p) + (1 - y_true) * torch.log(1 - p)).sum()

def evaluate_ipd_metrics(
    model,
    trajectories: List[Dict],
    *,
    K: int,
    max_ep_len: int,
    device: torch.device,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    structure_state_idx: Tuple[int, int, int, int, int] = (0,1,2,3,4),  # indices in state for risk,error,delta,infinity,continuous
    print_report: bool = True,
) -> Dict[str, float]:
    """
    Evaluate IPD metrics for standard DecisionTransformer (without MO support).
    
    Args:
        model: DecisionTransformer model (without MO)
        trajectories: List of trajectory dicts with keys: observations, actions, rewards
        K: Context length
        max_ep_len: Maximum episode length
        device: Torch device
        state_mean: State normalization mean
        state_std: State normalization std
        structure_state_idx: Indices for risk, error, delta, infinity, continuous in state
        print_report: Whether to print evaluation report
        
    Returns:
        Dictionary with metrics: Acc.t=1, Acc.t>1, LL.t=1, LL.t>1, Cor-Time, RMSE-Time, Cor-Avg, RMSE-Avg
    """
    model.eval()
    state_mean_t = torch.as_tensor(state_mean, device=device, dtype=torch.float32)
    state_std_t  = torch.as_tensor(state_std,  device=device, dtype=torch.float32)

    y_t1,  p_t1  = [], []
    y_tgt, p_tgt = [], []

    per_time_truth,  per_time_pred  = {}, {}
    per_struct_truth, per_struct_pred = {}, {}

    with torch.no_grad():
        for traj in trajectories:
            S = traj["observations"].astype(np.float32)    # (T, state_dim)
            A = traj["actions"].astype(np.float32)         # (T, 1)  {0,1}
            R = traj["rewards"].astype(np.float32)         # (T,)

            T = S.shape[0]
            state_dim = S.shape[1]

            for si in range(T):
                tlen = min(K, T - si)

                s_i  = S[si:si+tlen]
                a_i  = A[si:si+tlen]
                r_i  = R[si:si+tlen]

                # pad to length K
                s_pad  = np.zeros((K - tlen, state_dim), dtype=np.float32)
                a_pad  = np.ones((K - tlen, 1), dtype=np.float32) * -10.0
                r_pad  = np.zeros((K - tlen, 1), dtype=np.float32)
                ts_pad = np.zeros((K - tlen,), dtype=np.int64)

                s_win  = np.concatenate([s_pad, s_i], axis=0)
                a_win  = np.concatenate([a_pad, a_i], axis=0)
                r_win  = np.concatenate([r_pad, r_i.reshape(-1,1)], axis=0)

                # simple RTG (γ=1)
                rtg = np.cumsum(r_i[::-1])[::-1].astype(np.float32)
                if rtg.shape[0] < tlen + 1:
                    rtg = np.concatenate([rtg, np.zeros((1,), dtype=np.float32)], 0)
                rtg = np.concatenate([np.zeros((K - tlen, 1), dtype=np.float32),
                                      rtg.reshape(-1,1)], 0)

                ts = np.arange(si, si + tlen, dtype=np.int64)
                ts = np.minimum(ts, max_ep_len - 1)
                ts = np.concatenate([ts_pad, ts], 0)

                mask = np.concatenate([np.zeros((K - tlen,)), np.ones((tlen,))], 0).astype(np.float32)

                # to tensors
                s_t   = torch.as_tensor(s_win, device=device, dtype=torch.float32)
                a_t   = torch.as_tensor(a_win, device=device, dtype=torch.float32)
                r_t   = torch.as_tensor(r_win, device=device, dtype=torch.float32)
                rtg_t = torch.as_tensor(rtg, device=device, dtype=torch.float32)
                ts_t  = torch.as_tensor(ts,  device=device, dtype=torch.long)
                m_t   = torch.as_tensor(mask,device=device, dtype=torch.float32)

                # normalize states as in training
                s_t = (s_t - state_mean_t) / state_std_t

                # model forward (standard DecisionTransformer without MO)
                _, a_pred, _ = model.forward(
                    s_t.unsqueeze(0), a_t.unsqueeze(0), None,
                    rtg_t.unsqueeze(0)[:, :-1], ts_t.unsqueeze(0),
                    attention_mask=m_t.unsqueeze(0)
                )

                # valid positions only
                a_pred = a_pred.reshape(K, -1)[m_t.bool()]           # (tlen, 1)
                y_true = a_t.reshape(K, -1)[m_t.bool()]              # (tlen, 1)

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
        print("IPD Eval (No MO) →", metrics_str)

    return report

