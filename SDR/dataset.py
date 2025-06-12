import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from SDR.utils import * 


class AugmentedDataset(Dataset):
    def __init__(self, data_dir, has_gt=True, transform=None):
        """
        Args:
            data_dir (str): dataset toot directory 
            transform (callable, optional): transformation to be applied on images. Defaults to None.
        """
        self.sample_dirs = sorted(glob(os.path.join(data_dir, '*/')))
        self.transform = transform 
        self.has_gt = has_gt

    def __len__(self):
        return len(self.sample_dirs)
    
    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        # Load data as np.ndarray
        rgb = load_png_as_npy(os.path.join(sample_dir, 'rgb.png'))
        sparse_depth = np.load(os.path.join(sample_dir, 'sparse_depth.npy'))
        if self.has_gt: 
            gt = np.load(os.path.join(sample_dir, 'gt.npy'))
        normal = np.load(os.path.join(sample_dir, 'normal.npy'))

        # Convert into torch tensors
        rgb = torch.from_numpy(rgb).permute(2,0,1).float()  # (3,H,W)
        sparse_depth = torch.from_numpy(sparse_depth).unsqueeze(0).float() # (1,H,W)
        if self.has_gt: 
            gt = torch.from_numpy(gt).unsqueeze(0).float() # (1,H,W)
        normal = torch.from_numpy(normal).permute(2,0,1).float() # (3,H,W)

        if not self.has_gt:
            return {
                'rgb': rgb,
                'sparse_depth': sparse_depth,
                'normal': normal
            }
        else: 
            return {
                'rgb': rgb,
                'sparse_depth': sparse_depth,
                'gt': gt,
                'normal': normal
            }


def load_dataset(
    data_dir='./data/augmentation/',
    batch_size=8,
    has_gt=True, 
):
    """
    prepare dataset loader with batch 

    Args:
        data_dir (str, optional): dataset root directory.
        batch_size (int, optional): batch size. 
    
    Returns: 
        dataloader (torch.utils.data.DataLoader): prepared dataloader
    """
    dataset = AugmentedDataset(data_dir, has_gt=has_gt)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


"""
Usage example

for batch in dataloader:
    rgb = batch['rgb']              # (B, 3, H, W)
    sparse = batch['sparse_depth']  # (B, 1, H, W)
    gt = batch['gt']                # (B, 1, H, W)
    normal = batch['normal']        # (B, 3, H, W) 
    break
"""
