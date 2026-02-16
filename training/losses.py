import torch.nn.functional as F
import torch

def reconstruction_loss(pred, target):
    return F.mse_loss(pred, target)

def rollout_loss(pred_seq, gt_seq):
    """
    Penalize drift during multi-step prediction
    """
    return F.mse_loss(pred_seq, gt_seq)

def spatial_smoothness_loss(S):
    """
    Encourage spatially realistic radio maps.
    Penalizes sharp discontinuities.
    """
    dx = torch.abs(S[:, :, :, 1:] - S[:, :, :, :-1]).mean()
    dy = torch.abs(S[:, :, 1:, :] - S[:, :, :-1, :]).mean()
    return dx + dy

def total_loss(
    one_step_pred,
    one_step_gt,
    rollout_pred,
    rollout_gt,
    λ_rollout=0.5,
    λ_smooth=0.05
):
    L1 = reconstruction_loss(one_step_pred, one_step_gt)
    Lr = rollout_loss(rollout_pred, rollout_gt)
    Ls = spatial_smoothness_loss(rollout_pred)

    return L1 + λ_rollout * Lr + λ_smooth * Ls

