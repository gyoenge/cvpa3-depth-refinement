import torch
import torch.nn as nn
import torch.nn.functional as F
# from tqdm import tqdm


class HoleFiller(nn.Module):
    def __init__(self, kernel_size=(7, 7), iter=50):
        """
        Hole filling module using convolution-based smoothing on sparse depth.

        Args:
            kernel_size (tuple): e.g., (3, 3)
            iter (int): number of smoothing iterations
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.iter = iter

        # Define convolution kernel (averaging kernel)
        kernel = torch.ones(1, 1, *kernel_size)
        self.register_buffer('kernel', kernel)

    def forward(self, sparse_depth):
        """
        Args:
            sparse_depth (torch.Tensor): (B, 1, H, W) torch tensor with zeros in holes

        Returns:
            torch.Tensor: filled depth map (B, 1, H, W)
        """
        assert sparse_depth.dim() == 4 and sparse_depth.size(1) == 1, \
            "Input must be of shape (B, 1, H, W)"

        filled = sparse_depth.clone()
        B, _, H, W = filled.shape
        pad_h = (self.kernel_size[0] - 1) // 2
        pad_w = (self.kernel_size[1] - 1) // 2

        # for _ in tqdm(range(self.iter), desc="Hole Filling..."):
        for _ in range(self.iter):
            mask = (filled != 0).float()  # 1 where valid, 0 where hole
            padded_filled = F.pad(filled, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
            padded_mask = F.pad(mask, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)

            sum_conv = F.conv2d(padded_filled, self.kernel, padding=0)
            count_conv = F.conv2d(padded_mask, self.kernel, padding=0)

            avg_conv = torch.where(count_conv > 0, sum_conv / count_conv, torch.zeros_like(sum_conv))

            filled = torch.where(mask.bool(), filled, avg_conv)

        return filled

