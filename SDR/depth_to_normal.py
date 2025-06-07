import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class Depth2Normal(nn.Module):
    """
    Module to convert depth map to surface normals using intrinsic matrix.

    Args:
        K (torch.Tensor): Intrinsic matrix of shape (1, 3, 3)

    Input:
        depth (torch.Tensor): Depth map of shape (B, 1, H, W)

    Output:
        point_map (torch.Tensor): 3D point map of shape (B, 3, H, W)
        normals (torch.Tensor): Normal map of shape (B, 3, H, W)
    """

    def __init__(self, K: torch.Tensor):
        super().__init__()
        assert K.shape == (1, 3, 3), f"Expected K shape (1, 3, 3), got {K.shape}"
        self.register_buffer(
            'K_inv',
            torch.inverse(K)
        )

    def forward(self, depth: torch.Tensor):
        B, _, H, W = depth.shape
        device = depth.device

        # Generate pixel coordinates
        ys = torch.arange(0, H, device=device)
        xs = torch.arange(0, W, device=device)
        ygrid, xgrid = torch.meshgrid(ys, xs, indexing='ij')
        ones = torch.ones_like(xgrid)
        pix_coords = torch.stack([xgrid, ygrid, ones], dim=0).float()  # (3, H, W)
        pix_coords = pix_coords.unsqueeze(0).repeat(B, 1, 1, 1)        # (B, 3, H, W)

        # Backproject to 3D
        pix_coords_flat = pix_coords.view(B, 3, -1)                    # (B, 3, H*W)
        cam_coords = torch.bmm(self.K_inv, pix_coords_flat)                # (B, 3, H*W)
        cam_coords = cam_coords.view(B, 3, H, W) * (-depth)           # (B, 3, H, W)
        point_map = cam_coords 

        # Compute surface normals via cross product
        point_map_pad = F.pad(point_map, (0, 1, 0, 1), mode='replicate')  # (B, 3, H+1, W+1)
        dx = point_map_pad[:, :, :-1, 1:] - point_map_pad[:, :, :-1, :-1]
        dy = point_map_pad[:, :, 1:, :-1] - point_map_pad[:, :, :-1, :-1]

        normals = torch.cross(dx, dy, dim=1)
        normals = F.normalize(normals, dim=1)

        return point_map, normals

