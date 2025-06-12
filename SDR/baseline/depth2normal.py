import torch
import torch.nn as nn
import torch.nn.functional as F


class Depth2Normal(nn.Module):
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
        K_inv = self.K_inv.repeat(B, 1, 1)
        cam_coords = torch.bmm(K_inv, pix_coords_flat)                # (B, 3, H*W)
        point_map = cam_coords.view(B, 3, H, W) * (-depth)            # (B, 3, H, W)

        # Compute surface normals via cross product
        ##### (1) initial
        # point_map_pad = F.pad(point_map, (0, 1, 0, 1), mode='replicate')  # (B, 3, H+1, W+1)
        # dx = point_map_pad[:, :, :-1, 1:] - point_map_pad[:, :, :-1, :-1]
        # dy = point_map_pad[:, :, 1:, :-1] - point_map_pad[:, :, :-1, :-1]
        ##### (2) central difference method 
        point_map_pad = F.pad(point_map, (1, 1, 1, 1), mode='replicate')  # (B, 3, H+2, W+2)
        dx = (point_map_pad[:, :, 1:-1, 2:] - point_map_pad[:, :, 1:-1, :-2]) / 2  # (B, 3, H, W)
        dy = (point_map_pad[:, :, 2:, 1:-1] - point_map_pad[:, :, :-2, 1:-1]) / 2
        ##### (3) Use Sobel Kernels (for each channel)
        # sobel_x = torch.tensor([[[-1., 0., 1.],
        #                         [-2., 0., 2.],
        #                         [-1., 0., 1.]]]).expand(3, 1, 3, 3)
        # sobel_y = torch.tensor([[[-1., -2., -1.],
        #                         [ 0.,  0.,  0.],
        #                         [ 1.,  2.,  1.]]]).expand(3, 1, 3, 3)
        # sobel_x = torch.tensor([[[-1., 0., 1.],
        #                         [-1.5, 0., 1.5],
        #                         [-1., 0., 1.]]]).expand(3, 1, 3, 3) / 6.0
        # sobel_y = torch.tensor([[[-1., -1.5, -1.],
        #                         [ 0.,  0.,   0.],
        #                         [ 1.,  1.5,  1.]]]).expand(3, 1, 3, 3) / 6.0
        # dx = F.conv2d(point_map, sobel_x.to(point_map.device), padding=1, groups=3)
        # dy = F.conv2d(point_map, sobel_y.to(point_map.device), padding=1, groups=3)
        # sobel_x = torch.tensor([[
        #     [-1.,-1. ]
        # ]])
        # groups=3는 채널 별로 필터를 독립적으로 적용하기 위해 필요합니다 (3D point 기준)
        ##### (4) (5,5) Sobel 
        # 5x5 Sobel kernel in x and y directions (approximation)
        # sobel_x = torch.tensor([[[
        #     [-2, -1, 0, 1, 2],
        #     [-3, -2, 0, 2, 3],
        #     [-4, -3, 0, 3, 4],
        #     [-3, -2, 0, 2, 3],
        #     [-2, -1, 0, 1, 2]
        # ]]], dtype=torch.float32).expand(3, 1, 5, 5) / 48.0  # Normalize
        # sobel_y = torch.tensor([[[
        #     [-2, -3, -4, -3, -2],
        #     [-1, -2, -3, -2, -1],
        #     [ 0,  0,  0,  0,  0],
        #     [ 1,  2,  3,  2,  1],
        #     [ 2,  3,  4,  3,  2]
        # ]]], dtype=torch.float32).expand(3, 1, 5, 5) / 48.0
        # dx = F.conv2d(point_map, sobel_x.to(point_map.device), padding=2, groups=3)
        # dy = F.conv2d(point_map, sobel_y.to(point_map.device), padding=2, groups=3)

        normals = torch.cross(dx, dy, dim=1)
        normals = F.normalize(normals, dim=1, eps=1e-6)

        return normals

        ##### (5) Candidate Method
        # candidates = []
        # offsets = [1, 2, 3]
        # # padding 충분히 수행
        # pad = max(offsets)
        # pad_total = pad + max(offsets)
        # point_map_pad = F.pad(point_map, (pad_total, pad_total, pad_total, pad_total), mode='replicate')  # (B, 3, H+2p, W+2p)
        # H, W = point_map.shape[-2:]
        # start_h = pad_total
        # start_w = pad_total
        # for o in offsets:
        #     dx = point_map_pad[:, :, start_h:start_h+H, start_w+o:start_w+o+W] - \
        #         point_map_pad[:, :, start_h:start_h+H, start_w:start_w+W]

        #     dy = point_map_pad[:, :, start_h+o:start_h+o+H, start_w:start_w+W] - \
        #         point_map_pad[:, :, start_h:start_h+H, start_w:start_w+W]
        #     n = torch.cross(dx, dy, dim=1)
        #     n = F.normalize(n, dim=1, eps=1e-6)
        #     candidates.append(n)
        # normals_stack = torch.stack(candidates, dim=-1)  # (B, 3, H, W, num_candidates)

        # 1)
        # norms = torch.norm(normals_stack, dim=1)         # (B, H, W, num_candidates)
        # valid_mask = (norms < 2.0) & torch.isfinite(norms)
        # valid_count = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, 480, 640, 1)
        # valid_count = valid_count.permute(0, 3, 1, 2) # (B, 1, H, W)
        # normals_stack = normals_stack * valid_mask.unsqueeze(1) 
        # # print("normals_stack.shape: ", normals_stack.shape)
        # # print("valid_count.shape: ", valid_count.shape)
        # normals_mean = normals_stack.sum(dim=-1) / valid_count  # (B, 3, H, W)
        # normals_mean = F.normalize(normals_mean, dim=1, eps=1e-6)

        # 2) 
        # mean_normal = F.normalize(normals_stack.mean(dim=-1), dim=1, eps=1e-6)  # (B, 3, H, W)
        # cos_sim = (normals_stack * mean_normal.unsqueeze(-1)).sum(dim=1)  # (B, H, W, N)
        # valid_mask = (cos_sim >= 0.9) & torch.isfinite(cos_sim)
        # valid_count = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, H, W, 1)
        # valid_count = valid_count.permute(0, 3, 1, 2)  # (B, 1, H, W)
        # normals_stack = normals_stack * valid_mask.unsqueeze(1)  # (B, 3, H, W, N)
        # normals_mean = normals_stack.sum(dim=-1) / valid_count  # (B, 3, H, W)
        # normals_mean = F.normalize(normals_mean, dim=1, eps=1e-6)

        return normals_mean

