import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothHoleFiller(nn.Module):
    def __init__(self, kernel_size=7, iter=50):
        """
        Hole filling module using convolution-based smoothing on sparse depth.

        Args:
            kernel_size (int): size of square kernel. e.g., 7 means 7x7 kernel
            iter (int): number of smoothing iterations
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.iter = iter

        # Define gaussian kernel (smooth averaging kernel)
        std = kernel_size/6
        gaussian1d = torch.signal.windows.gaussian(kernel_size, std=std)
        gaussian2d = torch.outer(gaussian1d, gaussian1d) 
        gaussian2d /= gaussian2d.sum()
        kernel = gaussian2d.unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel', kernel)   # (1, 1, kernel_size, kernel_size)

    def forward(self, sparse_depth):
        """
        Args:
            sparse_depth (torch.Tensor): (B, 1, H, W) torch tensor with zeros in holes

        Returns:
            torch.Tensor: filled depth map (B, 1, H, W)
        """
        assert sparse_depth.dim() == 4 and sparse_depth.size(1) == 1, \
            "Input must be of shape (B, 1, H, W)"

        sparse = sparse_depth.clone()
        filled = sparse_depth.clone()
        B, _, H, W = filled.shape
        pad_h = (self.kernel_size - 1) // 2
        pad_w = (self.kernel_size - 1) // 2

        for i in range(self.iter):
            mask = (filled != 0).float()  # 1 where valid, 0 where hole
            padded_filled = F.pad(filled, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
            padded_mask = F.pad(mask, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)

            sum_conv = F.conv2d(padded_filled, self.kernel, padding=0)
            count_conv = F.conv2d(padded_mask, self.kernel, padding=0)

            avg_conv = torch.where(count_conv > 0, sum_conv / count_conv, torch.zeros_like(sum_conv))
            
            # update repeatedly except original sparse
            filled = torch.where(sparse!=0, sparse, avg_conv)

        return filled

