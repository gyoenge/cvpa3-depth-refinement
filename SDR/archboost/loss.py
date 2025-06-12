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
    # L1 Loss 
    loss_l1 = F.l1_loss(pred_normal, gt_normal)
    # loss_l1 = F.smooth_l1_loss(pred_normal, gt_normal)

    # # Cosine Similarity Loss
    # cos_sim = F.cosine_similarity(pred_normal, gt_normal, dim=1)  # (B, H, W)
    # # loss_cosine = (1.0 - cos_sim).mean()

    # # Degree Penalty Loss 
    # angle_threshold_deg=30
    # angle_rad = torch.acos(torch.clamp(cos_sim, -1 + 1e-6, 1 - 1e-6))  # (B, H, W)
    # angle_deg = angle_rad * (180.0 / torch.pi)
    # penalty_mask = (angle_deg > angle_threshold_deg).float()
    # loss_penalty = (penalty_mask * angle_deg / 90.0).mean()  # normalize by max 90 degrees

    # Edge Wegihted L1 Loss
    # loss_l1 = F.l1_loss(pred_normal, gt_normal, reduction='none')  # (B, 3, H, W)
    # edge_mask = compute_edge_mask_from_normal(gt_normal)  # (B, 1, H, W)
    # alpha = 4.0  # edge 5x weight
    # weight = 1.0 + alpha * edge_mask  # (B, 1, H, W)
    # weight = weight.expand_as(loss_l1)  # (B, 3, H, W)
    # weighted_loss = (loss_l1 * weight).mean()

    ## Edge Consistency Loss
    normal_pred_edge = compute_edge_mask_from_normal(pred_normal)
    normal_edge = compute_edge_mask_from_normal(gt_normal)
    # edge_consistency_loss = F.l1_loss(depth_edge, normal_edge)
    # edge_consistency_loss = F.l1_loss(normal_pred_edge, normal_edge)
    diff = torch.abs(normal_pred_edge - normal_edge)
    threshold=0.2
    mask = (diff > threshold).float()
    edge_consistency_loss = (diff * mask).sum() / (mask.sum() + 1e-6)

    return loss_l1 #+ 0.5 * edge_consistency_loss


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
    # dx = torch.abs(pred_depth[:, :, :, :-1] - pred_depth[:, :, :, 1:])
    # dy = torch.abs(pred_depth[:, :, :-1, :] - pred_depth[:, :, 1:, :])
    eps = 1e-6
    dx = torch.abs(torch.log(pred_depth[:, :, :, :-1] + eps) - torch.log(pred_depth[:, :, :, 1:] + eps))
    dy = torch.abs(torch.log(pred_depth[:, :, :-1, :] + eps) - torch.log(pred_depth[:, :, 1:, :] + eps))
    mask_dx = edge_mask[:, :, :, :-1]
    mask_dy = edge_mask[:, :, :-1, :]
    weight_dx = 1.0 - mask_dx  # penalize only non-edge
    weight_dy = 1.0 - mask_dy
    edgeaware_smoothness_loss = (dx * weight_dx).mean() + (dy * weight_dy).mean()
    # weight_dx, weight_dy = compute_soft_edge_weight_from_normal(gt_normal)
    # loss_dx = (dx * weight_dx[:, :, :, :-1]).sum() / (weight_dx[:, :, :, :-1].sum() + eps)
    # loss_dy = (dy * weight_dy[:, :, :-1, :]).sum() / (weight_dy[:, :, :-1, :].sum() + eps)
    # edgeaware_smoothness_loss = loss_dx + loss_dy

    ## Edge Consistency Loss
    # depth_edge = compute_depth_edge(pred_depth)
    # normal_edge = compute_edge_mask_from_normal(gt_normal)
    # # edge_consistency_loss = F.l1_loss(depth_edge, normal_edge)
    # edge_consistency_loss = F.l1_loss(depth_edge * normal_edge, normal_edge, reduction='sum')  \
    #      / (normal_edge.sum() + 1e-6)

    return penalty_minus + edgeaware_smoothness_loss # + 0.5 * edge_consistency_loss

### Aux Helpers

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

def compute_soft_edge_weight_from_normal(normals):
    """
    Compute soft edge weights based on normal angular differences.
    normals: (B, 3, H, W)
    """

    # x-gradient (W 방향)
    grad_x = normals[:, :, :, 1:] - normals[:, :, :, :-1]
    grad_x = F.pad(grad_x, pad=(0,1,0,0), mode='replicate')  # pad W-right by 1

    # y-gradient (H 방향)
    grad_y = normals[:, :, 1:, :] - normals[:, :, :-1, :]
    grad_y = F.pad(grad_y, pad=(0,0,0,1), mode='replicate')  # pad H-bottom by 1

    mag_x = torch.norm(grad_x, dim=1, keepdim=True)  # (B, 1, H, W)
    mag_y = torch.norm(grad_y, dim=1, keepdim=True)

    weight_x = torch.exp(-mag_x * 10.0)
    weight_y = torch.exp(-mag_y * 10.0)

    return weight_x, weight_y

def dilate_edge_mask(edge_mask: torch.Tensor, kernel_size=3):
    """
    edge_mask: (B, 1, H, W), binary {0,1} mask
    Return: dilated mask where 3x3 neighborhood contains any edge → 1
    """
    assert edge_mask.dim() == 4 and edge_mask.size(1) == 1, "Expected shape (B,1,H,W)"

    # 3x3 커널로 max pooling → 커널 안에 하나라도 1이 있으면 1
    dilated = F.max_pool2d(edge_mask, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    return dilated

def compute_depth_edge(depth):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=depth.device).unsqueeze(0).unsqueeze(0)
    sobel_y = sobel_x.transpose(2, 3)

    edge_x = F.conv2d(depth, sobel_x, padding=1)
    edge_y = F.conv2d(depth, sobel_y, padding=1)
    edge_mag = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)  # (B, 1, H, W)
    return edge_mag