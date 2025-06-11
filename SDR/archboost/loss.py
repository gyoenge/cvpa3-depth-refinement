import torch
import torch.nn.functional as F

def acrhboost_loss(
    # predicted 
    pred_depth, pred_normal, 
    # ground truth 
    gt_sparse, gt_normal,
    # hyperparameter 
    alpha=1.0, beta=0.2,
    lambda_aux=0.1 
): 
    loss_sparse = sparse_loss(pred_depth, gt_sparse)
    loss_normal = normal_loss(pred_normal, gt_normal)
    loss_aux = auxiliary_depth_loss(pred_depth)
    loss = alpha * loss_sparse + beta * loss_normal + lambda_aux * loss_aux
    return {
        'total': loss,
        'sparse': loss_sparse,
        'normal': loss_normal,
        'aux': loss_aux,
    }
    
"""
loss modules
"""

def sparse_loss(pred_depth, sparse_depth):
    """MSE Loss on sparse points only"""
    mask_sparse = (sparse_depth > 0).float()
    loss = F.mse_loss(pred_depth * mask_sparse, sparse_depth, reduction='sum')
    return loss / (mask_sparse.sum() + 1e-8)

def normal_loss(pred_normal, gt_normal):
    """L1 Loss between predicted and ground-truth normal"""
    return F.l1_loss(pred_normal, gt_normal)

def auxiliary_depth_loss(pred_depth):
    """Smoothness Loss for predicted depth"""
    dx = torch.abs(pred_depth[:, :, :, :-1] - pred_depth[:, :, :, 1:])  # (B, 1, H, W-1)
    dy = torch.abs(pred_depth[:, :, :-1, :] - pred_depth[:, :, 1:, :])  # (B, 1, H-1, W)
    return dx.mean() + dy.mean()