import os 
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import random
from SDR.utils import *


def augmentation(
    ### config
    SOURCE_DIR = './data/train/',
    DEST_DIR = './data/augmentation/',
    AUG_COUNT = 1000,
    CROP_SCALE = 0.4,
    SPARSE_RATE = 0.2,
):
    """
    Data Augmentaion : Augment single train sample into multiple diverse variations
    """
    ### file exist validation check 
    if not os.path.exists(SOURCE_DIR):
        raise FileNotFoundError(f"Source directory {SOURCE_DIR} does not exist.")
    rgb_path = os.path.join(SOURCE_DIR, 'rgb.png')
    depth_path = os.path.join(SOURCE_DIR, 'gt.npy') 
    normal_path = os.path.join(SOURCE_DIR, 'normal.npy') 
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"File RGB {rgb_path} does not exist.")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"File Depth {depth_path} does not exist.")
    if not os.path.exists(normal_path):
        raise FileNotFoundError(f"File Normal {normal_path} does not exist.")
    
    ### mkdir & load 
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    os.makedirs(DEST_DIR, exist_ok=True)
    rgb = load_png_as_npy(rgb_path)  # (H,W,3)
    depth = np.load(depth_path)      # (H,W) 
    normal = np.load(normal_path)    # (H,W)

    ### data shape validation check 
    if (rgb.shape[:2] != depth.shape[:2]) or (rgb.shape[:2] != normal.shape[:2]):
        raise ValueError(f"Mismatch between rgb {rgb.shape[:2]}, depth {depth.shape}, and normal {normal.shape}")
    else: 
        print(f"Successfully loaded original (rgb, depth, normal) data")
        print(f"| with each shape : {rgb.shape}, {depth.shape}, {normal.shape}")
 
    ### augmentation
    augmentor = Augmentor(
        crop_scale=CROP_SCALE,
        sparse_rate=SPARSE_RATE
    )
    for aug_idx in tqdm(range(1, AUG_COUNT+1), desc="Augmentation..."):
        aug_rgb, aug_depth, aug_normal = augmentor.random_convert(rgb, depth, normal)
        aug_sparse_depth = augmentor.make_sparse_depth(aug_depth)
        augmentor.save_augmented_sample(
            dest_dir=DEST_DIR, 
            idx=aug_idx, 
            rgb=aug_rgb, 
            depth=aug_depth, 
            normal=aug_normal, 
            sparse_depth=aug_sparse_depth
        )
    print(f"| Augmentation -- CROP_SCALE : {CROP_SCALE}, CROPPED_SIZE : {aug_depth.shape}")
    print(f"| Augmentation -- SPARSE_RATE : {SPARSE_RATE}")


class Augmentor:
    def __init__(self, crop_scale=0.4, sparse_rate=0.2):
        self.crop_scale = crop_scale
        self.sparse_rate = sparse_rate

    def random_convert(self, rgb, depth, normal):
        """
        Apply the same random crop and flip to rgb, depth, normal.
        Additionally apply Gaussian noise to rgb only.

        Args:
            rgb (np.ndarray): RGB image (H, W, 3)
            depth (np.ndarray): depth map (H, W)
            normal (np.ndarray): normal map (H, W)

        Returns:
            tuple: (rgb_aug, depth_aug, normal_aug)
        """
        H, W = rgb.shape[:2]
        crop_h, crop_w = int(H * self.crop_scale), int(W * self.crop_scale)

        # Random Crop
        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)

        rgb = rgb[top:top + crop_h, left:left + crop_w]
        depth = depth[top:top + crop_h, left:left + crop_w]
        normal = normal[top:top + crop_h, left:left + crop_w]

        # Random Horizontal Flip
        if random.random() < 0.5:
            rgb = np.fliplr(rgb)
            depth = np.fliplr(depth)
            normal = np.fliplr(normal)

        # Gaussian Noise (RGB only)
        if random.random() < 0.5:
            noise = np.random.normal(0, 10, rgb.shape).astype(np.float32)
            rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return rgb, depth, normal

    def save_augmented_sample(self, dest_dir, idx, rgb, depth, normal, sparse_depth, logging=False):
        """
        Save augmented sample to DEST_DIR/0000idx/ with proper filenames.

        Args:
            idx (np.ndarray): _description_
            rgb (np.ndarray): RGB image (H, W, 3)
            depth (np.ndarray): depth map (H, W)
            normal (np.ndarray): normal map (H, W)
            sparse_depth (np.ndarray): (H, W)
        """
        sample_dir = os.path.join(dest_dir, f"{idx:04d}")
        os.makedirs(sample_dir, exist_ok=True)

        # Save RGB image as PNG
        Image.fromarray(rgb).save(os.path.join(sample_dir, "rgb.png"))

        # Save depth and normal as .npy
        np.save(os.path.join(sample_dir, "gt.npy"), depth)
        np.save(os.path.join(sample_dir, "normal.npy"), normal)

        # Save spase depth as .npy
        np.save(os.path.join(sample_dir, "sparse_depth.npy"), sparse_depth)

        # Save log
        if logging: 
            print(f"Successfully saved augmented files..")
            print(f"| rgb : {os.path.join(sample_dir, "rgb.png")}")
            print(f"| depth : {os.path.join(sample_dir, "gt.npy")}")
            print(f"| normal : {os.path.join(sample_dir, "normal.npy")}")
            print(f"| sparse_depth : {os.path.join(sample_dir, "sparse_depth.npy")}")

        # Save visualization images
        plt.imsave(os.path.join(sample_dir, "gt.png"), depth, cmap='jet')
        normal_vis = (normal + 1) / 2  # [-1, 1] → [0, 1]
        normal_vis = np.clip(normal_vis, 0, 1)
        plt.imsave(os.path.join(sample_dir, "normal.png"), normal_vis)
        plt.imsave(os.path.join(sample_dir, "sparse_depth.png"), sparse_depth, cmap='jet')

        # Save log 
        if logging: 
            print(f"Also saved visualized .npy files..")
            print(f"| depth : {os.path.join(sample_dir, "gt.png")}")
            print(f"| normal : {os.path.join(sample_dir, "normal.png")}")
            print(f"| sparse_depth : {os.path.join(sample_dir, "sparse_depth.png")}")

    def make_sparse_depth(self, depth):
        """
        Create sparse depth map by randomly sampling a portion of valid (non-zero) pixels.
        
        Args:
            depth (np.ndarray): (H, W) full dense depth

        Returns:
            np.ndarray: sparse depth map of same shape
        """
        sparse_depth = np.zeros_like(depth)
        valid_mask = depth > 0
        valid_indices = np.argwhere(valid_mask)
        
        sample_size = int(len(valid_indices) * self.sparse_rate)
        sampled_indices = valid_indices[np.random.choice(len(valid_indices), sample_size, replace=False)]

        for y, x in sampled_indices:
            sparse_depth[y, x] = depth[y, x]

        return sparse_depth



if __name__ == '__main__':
    augmentation()