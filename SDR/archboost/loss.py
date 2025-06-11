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
    loss_aux = auxiliary_depth_loss(pred_depth, gt_normal)
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
    ## Cosine similarity loss? 
    # cos = F.cosine_similarity(pred_normal, gt_normal, dim=1)
    # return 1.0 - cos.mean()

def auxiliary_depth_loss(pred_depth, gt_normal):
    """Auxiliary Loss for predicted depth"""
    ## Smoothness Loss 
    # dx = torch.abs(pred_depth[:, :, :, :-1] - pred_depth[:, :, :, 1:])  # (B, 1, H, W-1)
    # dy = torch.abs(pred_depth[:, :, :-1, :] - pred_depth[:, :, 1:, :])  # (B, 1, H-1, W)
    # smoothness_loss = dx.mean() + dy.mean()

    ## Minus Penalty Loss (Ensure Nonzero)
    penalty_minus = F.relu(0.0 - pred_depth).mean()

    ## Edge-aware Smoothness Loss
    edge_mask = compute_edge_mask_from_normal(gt_normal)
    dx = torch.abs(pred_depth[:, :, :, :-1] - pred_depth[:, :, :, 1:])
    dy = torch.abs(pred_depth[:, :, :-1, :] - pred_depth[:, :, 1:, :])
    mask_dx = edge_mask[:, :, :, :-1]
    mask_dy = edge_mask[:, :, :-1, :]
    weight_dx = 1.0 - mask_dx  # penalize only non-edge
    weight_dy = 1.0 - mask_dy
    edgeaware_smoothness_loss = (dx * weight_dx).mean() + (dy * weight_dy).mean()
    
    return penalty_minus + edgeaware_smoothness_loss 


def compute_edge_mask_from_normal(gt_normal):
    """
    Args:
        gt_normal: (B, 3, H, W), float32 in [-1, 1]
    Returns:
        edge_mask: (B, 1, H, W), float32 in [0, 1], 1 at edges
    """
    dx = torch.abs(gt_normal[:, :, :, 1:] - gt_normal[:, :, :, :-1])
    dy = torch.abs(gt_normal[:, :, 1:, :] - gt_normal[:, :, :-1, :])
    edge_x = dx.mean(dim=1, keepdim=True)  # (B, 1, H, W-1)
    edge_y = dy.mean(dim=1, keepdim=True)  # (B, 1, H-1, W)
    edge_mask = F.pad(edge_x, (0,1,0,0)) + F.pad(edge_y, (0,0,0,1))  # (B, 1, H, W)
    edge_mask = (edge_mask > 0.1).float()  # threshold; you can tune this
    return edge_mask