import torch 
import torch.nn as nn
from SDR.baseline.holefiller import HoleFiller
from SDR.baseline.unet import UNet
from SDR.baseline.depth2normal import Depth2Normal

class Baseline(nn.Module):
    def __init__(self,
        holefilling_kernel_size=7,
        holefilling_iter=50,   
    ):
        super().__init__()
        self.holefiller = HoleFiller(
            kernel_size=holefilling_kernel_size, 
            iter=holefilling_iter
        )
        self.rgbd2depth = UNet(
            n_channels=4, 
            n_classes=1, 
            bilinear=False 
        )
        self.depth2normal = Depth2Normal(
            K=torch.tensor([[
                [5.1885790117450188e+02, 0.0, 3.2558244941119034e+02],
                [0.0, 5.1946961112127485e+02, 2.5373616633400465e+02],
                [0.0, 0.0, 1.0]
            ]], dtype=torch.float32)  # (1, 3, 3)
        )
        
    def forward(self,
        rgb, # (B, 3, H, W)
        sparse_depth # (B, 1, H, W)
    ):
        # Hole filling
        initial_depth = self.holefiller(sparse_depth) # (B, 1, H, W)

        # RGBD to Depth 
        rgbd = torch.cat([rgb, initial_depth], dim=1) # (B, 4, H, W)
        depth = self.rgbd2depth(rgbd) # (B, 1, H, W)

        # Depth to Normal 
        point_map, normals = self.depth2normal(depth) # (B, 3, H, W), (B, 3, H, W)
        
        return initial_depth, depth, normals # (B, 1, H, W), (B, 3, H, W)
    