import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image


"""
load
"""
def load_png_as_npy(png_path='./data/train_latest/rgb.png'):
    image = Image.open(png_path).convert('RGB')
    image_np = np.array(image) 
    return image_np

"""
save
"""
def save_depth_image(depth_np: np.ndarray, save_path: str):
    """
    Save a depth map as a colormap image.

    Args:
        depth_np (np.ndarray): Depth array of shape (H, W)
        save_path (str): Path to save the image (.png or .jpg)
    """
    plt.figure()
    plt.axis('off')
    im = plt.imshow(depth_np, cmap='jet')
    plt.colorbar(im, shrink=0.6)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_normal_image(normal_np: np.ndarray, save_path: str):
    """
    Save a normal map as an RGB image.

    Args:
        normal_np (np.ndarray): Normal array of shape (H, W, 3), range: [-1, 1] or unnormalized
        save_path (str): Path to save the image (.png or .jpg)
    """
    # normalize to [0, 1] for visualization
    normal_vis = (normal_np + 1.0) / 2.0  # from [-1,1] to [0,1]
    normal_vis = np.clip(normal_vis, 0, 1)

    plt.imsave(save_path, normal_vis)

"""
convert and save
"""
def save_depth_clip(
   root_dir: str     
): 
    """
    Convert predicted depth with clipping and Save.
    Conversion will be conducted for depth_pred.npy in all sub-directory of root.

    Args:
        root_dir (str): root directory path
    """
    sub_dirs = glob(os.path.join(root_dir, '*/'))

    for sub_dir in sub_dirs:
        depth_path = os.path.join(sub_dir, 'depth_pred.npy')
        if not os.path.exists(depth_path):
            print(f"depth_pred.npy not found in {sub_dir}, skipping.")
            continue

        # Load depth prediction
        depth = np.load(depth_path)

        # Clip the values
        min_depth = 0
        max_depth = 3
        depth_clipped = np.clip(depth, min_depth, max_depth)

        # Save as .npy
        # npy_save_path = os.path.join(sub_dir, 'depth_pred_clipped.npy')
        # np.save(npy_save_path, depth_clipped)

        # Save as .png
        png_save_path = os.path.join(sub_dir, 'depth_pred_clipped.png')
        save_depth_image(depth_clipped, png_save_path)

        print(f"Saved clipped depth to {sub_dir}")

def save_depth_outlier(
   root_dir: str     
): 
    """
    Detect up/down-outliers of predicted depth and Save.
    Conversion will be conducted for depth_pred.npy in all sub-directory of root.

    Args:
        root_dir (str): root directory path
    """
    sub_dirs = glob(os.path.join(root_dir, '*/'))

    for sub_dir in sub_dirs:
        depth_path = os.path.join(sub_dir, 'depth_pred.npy')
        if not os.path.exists(depth_path):
            print(f"depth_pred.npy not found in {sub_dir}, skipping.")
            continue

        # Load depth prediction
        depth = np.load(depth_path)
        H, W = depth.shape

        # Initialize RGB image as black
        outlier_rgb = np.zeros((H, W, 3), dtype=np.uint8)

        # Detect outliers 
        min_depth = 0
        max_depth = 3
        # Mask for min-outlier (sky blue)
        min_mask = depth < min_depth
        outlier_rgb[min_mask] = [135, 206, 235]  # sky blue (R,G,B)
        # Mask for max-outlier (red)
        max_mask = depth > max_depth
        outlier_rgb[max_mask] = [255, 0, 0]  # red

        # Save image
        # png_save_path = os.path.join(sub_dir, 'depth_pred_outlier.png')
        # plt.imsave(png_save_path, outlier_rgb)

        # Save image with legend using plt
        dpi = 100
        fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
        ax.imshow(outlier_rgb)
        ax.axis('off')
        legend_elements = [
            Patch(facecolor=(135/255, 206/255, 235/255), label='Down Outlier (< min)', edgecolor='gray'),
            Patch(facecolor=(255/255, 0, 0), label='Up Outlier (> max)', edgecolor='gray'),
            Patch(facecolor='black', label='Inlier', edgecolor='gray')
        ]
        # ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=9)
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9)
        png_save_path = os.path.join(sub_dir, 'depth_pred_outlier.png')
        plt.tight_layout()
        plt.savefig(png_save_path) #, bbox_inches='tight', pad_inches=0.2)
        plt.close()

        print(f"Saved outlier image to {png_save_path}")
