### import 
import os 
import logging 
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from SDR.hole_filling import hole_filling
from SDR.unet import UNet
from SDR.depth_to_normal import Depth2Normal
from SDR.utils import *

### config
# data
RGB_PATH = './data/train/rgb.png'
SPARSE_DEPTH_PATH = './data/train/sparse_depth.npy'
GT_DEPTH_PATH = './data/train/gt.npy'
GT_NORMAL_PATH = './data/train/normal.npy'
# hole filling 
HOLE_FILLING_ITER = 50
HOLE_FILLING_KERNEL_SIZE = (7,7)
# camera intrinsic
f_x = 5.1885790117450188e+02
f_y = 5.1946961112127485e+02
c_x = 3.2558244941119034e+02
c_y = 2.5373616633400465e+02
# train 
TRAIN_EPOCH = 1000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-1
ALPHA = 1.0 
BETA = 0.5#0.1
SAVE_DIR = './output/baseline'

### baseline 
def baseline():
    # load data 
    rgb = load_png_as_npy(png_path=RGB_PATH)  # (3, H, W)
    sparse_depth = np.load(SPARSE_DEPTH_PATH)  # (H, W)
    gt_depth = np.load(GT_DEPTH_PATH)  # (H, W)
    gt_normal = np.load(GT_NORMAL_PATH)  # (3, H, W)
    K_torch = torch.tensor([[
        [f_x,   0.0, c_x],
        [0.0,   f_y, c_y],
        [0.0,   0.0, 1.0]
    ]], dtype=torch.float32)  # (1, 3, 3)

    # hole filling [step 1]
    # np.ndarray(H,W) -> np.ndarray(H,W)
    print("Start hole filling...")
    initial_depth = hole_filling(
        sparse_depth=sparse_depth,
        iter=HOLE_FILLING_ITER,
        kernel_size=HOLE_FILLING_KERNEL_SIZE
    )
    print("End hole filling...")

    depth_initial_path = os.path.join(SAVE_DIR, "depth_initial.png")
    save_depth_image(initial_depth, depth_initial_path)
    print(f"Initial depth saved to {depth_initial_path}")

    # model : rgbd to normal [step 2,3]  
    # torch.Tensor(B,4,H,W) 
    # -> [depth] torch.Tensor(B,1,H,W)
    # -> [normal] torch.Tensor(B,3,H,W)
    class RGBD2Normal(nn.Module):
        def __init__(self, K):
            super().__init__()
            self.rgbd2depth = UNet(
                n_channels=4, 
                n_classes=1, 
                bilinear=False 
            )
            self.depth2normal = Depth2Normal(
                K=K
            )

        def forward(self, rgbd):
            depth = self.rgbd2depth(rgbd)
            point_map, normals = self.depth2normal(depth)
            return depth, normals 
        
    model = RGBD2Normal(K=K_torch)

    rgbd_np = np.concatenate((
        rgb, 
        np.expand_dims(initial_depth, axis=-1)
    ), axis=-1)  # (H, W. 4)
    rgbd = torch.from_numpy(rgbd_np).permute(2, 0, 1).unsqueeze(0).float() # (1, 4, H, W)

    # optimize [step 4] 
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Current device: {device}")
    model = model.to(device)
    rgbd = rgbd.to(device)
    optimizer = torch.optim.AdamW(
        model.rgbd2depth.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )

    gt_depth_torch = torch.from_numpy(sparse_depth).unsqueeze(0).unsqueeze(0).float() # (1,1,H,W)
    gt_depth_torch = gt_depth_torch.to(device)
    gt_normal_torch = torch.from_numpy(gt_normal).permute(2, 0, 1).unsqueeze(0).float() # (1,3,H,W)
    gt_normal_torch = gt_normal_torch.to(device)

    log_file_path = os.path.join(SAVE_DIR, 'log.txt')
    logging.basicConfig(
        filename=log_file_path,
        filemode='w', 
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    logging.info(f"[Hyperparameters] HOLE_FILLING_ITER={HOLE_FILLING_ITER}, HOLE_FILLING_KERNEL_SIZE={HOLE_FILLING_KERNEL_SIZE}, "
                 f"TRAIN_EPOCH={TRAIN_EPOCH}, LEARNING_RATE={LEARNING_RATE}, WEIGHT_DECAY={WEIGHT_DECAY}, "
                 f"ALPHA={ALPHA}, BETA={BETA}, ")
    logging.info("Start training...")

    for epoch in range(1,TRAIN_EPOCH+1):
        model.train()
        optimizer.zero_grad()

        depth, normals = model(rgbd)

        sparse_mask = (gt_depth_torch > 0)
        diff = (depth - gt_depth_torch)[sparse_mask]
        if diff.numel() > 0:
            loss_sparse = torch.sqrt(torch.mean(diff ** 2))  # masked RMSE
        else:
            loss_sparse = torch.tensor(0.0, device=depth.device)
        loss_normal = torch.sqrt(F.mse_loss(normals, gt_normal_torch))
        loss = ALPHA * loss_sparse + BETA * loss_normal 

        loss.backward()
        optimizer.step()

        logging.info(
            f"[Epoch {epoch}]"
            f"Loss Sparse: {loss_sparse:.4f}, Loss Normal: {loss_normal:.4f}, "
            f"Total Loss: {loss:.4f}"
        )

    # save results 
    model.eval()
    with torch.no_grad():
        rgbd = rgbd.to(device)
        depth_pred, normal_pred = model(rgbd)  # (1, 1, H, W), (1, 3, H, W)
    depth_pred_np = depth_pred.squeeze(0).squeeze(0).cpu().numpy()    # (H, W)
    normal_pred_np = normal_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    depth_pred_path = os.path.join(SAVE_DIR, "depth_pred.png")
    normal_pred_path = os.path.join(SAVE_DIR, "normal_pred.png")
    save_depth_image(depth_pred_np, depth_pred_path)
    logging.info(f"Predicted depth saved to {depth_pred_path}")
    save_normal_image(normal_pred_np, normal_pred_path)
    logging.info(f"Predicted normal saved to {normal_pred_path}")

    weight_path = os.path.join(SAVE_DIR, f"weight.pth")
    torch.save(model.state_dict(), weight_path)
    logging.info(f"Model weight saved to {weight_path}")


if __name__ == '__main__':
    baseline()
