import numpy as np 
from tqdm import tqdm 


def hole_filling(sparse_depth, iter=5, kernel_size=(3,3)):
    """
    Hole Filling - from sparse depth to initial depth 

    Parameters: 
        sparse_depth (np.ndarray): spase depth (H, W)
        iter (int): iteration number 
        kernel_size ((int,int)): kernel size

    Returns: 
        filled_depth (np.ndarray): hole-filled depth (H, W) 
    """
    h, w = sparse_depth.shape
    zero_indices = list(np.ndindex(sparse_depth.shape))
    kernel_h, kernel_w = kernel_size
    pad_h = (kernel_h - 1) // 2
    pad_w = (kernel_w - 1) // 2

    for i in tqdm(range(iter), desc="Hole Filling..."):
        # zero padding
        sparse_depth = np.pad(sparse_depth, 
                              pad_width=((pad_h, pad_h), (pad_w, pad_w)), 
                              mode='constant', constant_values=0)
        new_sparse_depth = np.zeros_like(sparse_depth)

        # convolution (for only holes) 
        for idx in zero_indices:
            idx_h, idx_w = idx
            surrondings = [value for value in sparse_depth[idx_h-pad_h:idx_h+pad_h+1, idx_w-pad_w:idx_w+pad_w+1].flatten() if value != 0]
            new_sparse_depth[idx_h, idx_w] = sum(surrondings) / len(surrondings) if len(surrondings) > 0 else 0

        # convert back to original shape
        sparse_depth = new_sparse_depth[pad_h:-pad_h, pad_w:-pad_w]
        
    return sparse_depth