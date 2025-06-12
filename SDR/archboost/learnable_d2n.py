import torch
import torch.nn as nn
import torch.nn.functional as F
from SDR.archboost.loss import compute_edge_mask_from_normal, dilate_edge_mask


class LearnableD2N(nn.Module):
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

        # self.refine = LearnableAvg()
        # self.refine = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(32),
        #     nn.Conv2d(32, 3, kernel_size=3, padding=1), 
        #     LearnableAvg()
        # )
        # self.refine = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, padding=1),     # Feature 확장
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(32),

        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),    # 더 깊게
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(64),

        #     nn.Conv2d(64, 32, kernel_size=3, padding=1),    # 다시 축소
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(32),

        #     nn.Conv2d(32, 3, kernel_size=3, padding=1),     # 최종 normal map (3채널)
        #     LearnableAvg(), 
        # )
        # self.refine = nn.Sequential(
        #     LearnableAvg(),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(3),
        #     LearnableAvg(),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(3),
        #     LearnableAvg(),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(3),
        #     LearnableAvg(), 
        # )


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
        # point_map_pad = F.pad(point_map, (0, 1, 0, 1), mode='replicate')  # (B, 3, H+1, W+1)
        # dx = point_map_pad[:, :, :-1, 1:] - point_map_pad[:, :, :-1, :-1]
        # dy = point_map_pad[:, :, 1:, :-1] - point_map_pad[:, :, :-1, :-1]
        point_map_pad = F.pad(point_map, (1, 1, 1, 1), mode='replicate')  # (B, 3, H+2, W+2)
        dx = (point_map_pad[:, :, 1:-1, 2:] - point_map_pad[:, :, 1:-1, :-2]) / 2  # (B, 3, H, W)
        dy = (point_map_pad[:, :, 2:, 1:-1] - point_map_pad[:, :, :-2, 1:-1]) / 2

        normals = torch.cross(dx, dy, dim=1)
        normals = F.normalize(normals, dim=1, eps=1e-6)

        # Light Learnable Layers 
        # if avg_pool: 
        # refined_normals = self.refine(normals)
        # refined_normals = F.avg_pool2d(normals, kernel_size=3, stride=1, padding=1)
        refined_normals = F.avg_pool2d(normals, kernel_size=5, stride=1, padding=2)
        refined_normals = F.normalize(refined_normals, dim=1, eps=1e-6)
        normals = refined_normals

        # Edge: Raw / Non-Edge: Refine
        # edge_mask = compute_edge_mask_from_normal(normals)
        # edge_mask = dilate_edge_mask(edge_mask)
        # refined_normals = edge_mask * normals + (1-edge_mask) * refined_normals

        return normals
    

class LearnableAvg(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_weight = nn.Parameter(torch.randn(3, 1, 3, 3))  # (out_channels, in_channels, kH, kW)

    def forward(self, x):
        weight = self.raw_weight.view(-1, 9)              # (9, 9)
        weight = F.softmax(weight, dim=1)                 # 각 커널 단위로 softmax
        weight = weight.view(3, 1, 3, 3)                  # 다시 Conv2d weight shape로
        return F.conv2d(x, weight, padding=1, groups=3)

