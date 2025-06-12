import torch
import torch.nn as nn
import torch.nn.functional as F


class AvgpoolD2N(nn.Module):
    """
    Module to convert depth map to surface normals using intrinsic matrix.

    Args:
        K (torch.Tensor): Intrinsic matrix of shape (1, 3, 3)

    Input:
        depth (torch.Tensor): Depth map of shape (B, 1, H, W)

    Output:
        normals (torch.Tensor): Normal map of shape (B, 3, H, W)
    """

    def __init__(self, 
        # default K for original train sample 
        K=torch.tensor([[
            [5.1885790117450188e+02, 0.0, 3.2558244941119034e+02],
            [0.0, 5.1946961112127485e+02, 2.5373616633400465e+02],
            [0.0, 0.0, 1.0]
        ]], dtype=torch.float32)   
    ):
        super().__init__()
        assert K.shape == (1, 3, 3), f"Expected K shape (1, 3, 3), got {K.shape}"
        self.register_buffer(
            'K_inv',
            torch.inverse(K)
        )

    def forward(self, 
                depth: torch.Tensor, 
                avg_pool=True, 
    ):
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
        K_inv = self.K_inv.repeat(B, 1, 1)
        cam_coords = torch.bmm(K_inv, pix_coords_flat)                # (B, 3, H*W)
        point_map = cam_coords.view(B, 3, H, W) * (-depth)            # (B, 3, H, W)

        # Compute surface normals via cross product
        ## (1)
        # point_map_pad = F.pad(point_map, (0, 1, 0, 1), mode='replicate')  # (B, 3, H+1, W+1)
        # dx = point_map_pad[:, :, :-1, 1:] - point_map_pad[:, :, :-1, :-1]
        # dy = point_map_pad[:, :, 1:, :-1] - point_map_pad[:, :, :-1, :-1]
        ## (2)
        point_map_pad = F.pad(point_map, (1, 1, 1, 1), mode='replicate')  # (B, 3, H+2, W+2)
        dx = (point_map_pad[:, :, 1:-1, 2:] - point_map_pad[:, :, 1:-1, :-2]) / 2  # (B, 3, H, W)
        dy = (point_map_pad[:, :, 2:, 1:-1] - point_map_pad[:, :, :-2, 1:-1]) / 2

        normals = torch.cross(dx, dy, dim=1)
        normals = F.normalize(normals, dim=1, eps=1e-6)

        # AvgPool 
        avgpool_kernel_size = 5
        refined_normals = F.avg_pool2d(normals, 
                                       kernel_size=avgpool_kernel_size, 
                                       stride=1, 
                                       padding=(avgpool_kernel_size-1)//2)
        refined_normals = F.normalize(refined_normals, dim=1, eps=1e-6)
        normals = refined_normals

        return normals
    

