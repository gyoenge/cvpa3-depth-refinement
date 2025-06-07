import matplotlib.pyplot as plt
import numpy as np
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

