import os
import logging
import math
import torch 
import torch.nn.functional as F
from SDR.dataset import load_dataset
from SDR.baseline.model import Baseline
from SDR.utils import *


def inference(
    data_dir='./data/orisample/',
    save_dir='./output/baseline/inference/',
    model_path='./data/model/baseline.pth',
    holefilling_kernel_size=7,
    holefilling_iter=50, 
    batch_size=8,
): 
    # save_dir setting 
    os.makedirs(save_dir, exist_ok=True)
    
    # logger setting 
    logging.basicConfig(
        filename=os.path.join(save_dir, 'result_log.txt'),
        filemode='w', 
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    logging.info(f"[Hyperparameters] holefilling_iter={holefilling_iter}, holefilling_kernel_size={holefilling_kernel_size}, "
                 f"batch_size={batch_size}")

    # device setting 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"[Device] {device}")

    # define dataloader 
    dataloader = load_dataset(
        data_dir=data_dir,
        batch_size=batch_size
    )

    # define model 
    model = Baseline(
        holefilling_kernel_size=holefilling_kernel_size,
        holefilling_iter=holefilling_iter,   
    )
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad(): 
        epoch_mse_initial = 0.0
        epoch_mse_depth = 0.0
        num_batches = 0
        for batch in dataloader:
            rgb = batch['rgb'].to(device)              # (B, 3, H, W)
            sparse = batch['sparse_depth'].to(device)  # (B, 1, H, W)
            gt = batch['gt'].to(device)                # (B, 1, H, W)
            normal = batch['normal'].to(device)        # (B, 3, H, W) 

            initial_depth, depth, normals = model(
                rgb, # (B, 3, H, W)
                sparse # (B, 1, H, W)
            )

            # RMSE log 
            mse_initial = F.mse_loss(initial_depth, gt, reduction='mean')
            mse_depth = F.mse_loss(depth, gt, reduction='mean')
            epoch_mse_initial += mse_initial.item()
            epoch_mse_depth += mse_depth.item()
            num_batches += 1
            logging.info(
                f"[Batch {num_batches}] "
                f"RMSE Initial: {math.sqrt(epoch_mse_initial):.4f}, "
                f"RMSE Depth: {math.sqrt(epoch_mse_depth):.4f}"
            )

            # save result 
            batch_len = sparse.size(0)
            for batch_idx in range(batch_len):
                sample_save_dir = os.path.join(save_dir, f"{num_batches}_{batch_idx}/")
                os.makedirs(sample_save_dir, exist_ok=True)
                # reuse last batch 
                depth_initial_np = initial_depth[batch_idx].squeeze(0).squeeze(0).cpu().numpy()    # (H, W)
                depth_pred_np = depth[batch_idx].squeeze(0).squeeze(0).cpu().numpy()    # (H, W)
                depth_gt_np = gt[batch_idx].squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
                depth_sparse_np = sparse[0].squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
                normal_pred_np = normals[0].squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
                normal_gt_np = normal[batch_idx].squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
                # vis results 
                save_depth_image(depth_initial_np, os.path.join(sample_save_dir, "depth_initial.png"))
                save_depth_image(depth_pred_np, os.path.join(sample_save_dir, "depth_pred.png"))
                save_depth_image(depth_gt_np, os.path.join(sample_save_dir, "depth_gt.png"))
                save_depth_image(depth_sparse_np, os.path.join(sample_save_dir, "depth_sparse.png"))
                save_normal_image(normal_pred_np, os.path.join(sample_save_dir, "normal_pred.png"))
                save_normal_image(normal_gt_np, os.path.join(sample_save_dir, "normal_gt.png"))
                # npy for submission
                np.save(os.path.join(sample_save_dir, "depth_initial.npy"), depth_initial_np)
                np.save(os.path.join(sample_save_dir, "depth_pred.npy"), depth_pred_np)
            logging.info(f"[Batch {num_batches}] Batch results saved into {os.path.join(save_dir,f"{num_batches}_*/")}")

    # end log
    logging.info(f"[END] Inference finished for entire dataset")
