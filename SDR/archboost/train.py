import os 
import logging
import math
import torch 
import torch.nn.functional as F
from SDR.dataset import load_dataset
from SDR.archboost.model import ArchBoost
from SDR.archboost.loss import acrhboost_loss
from SDR.utils import *

def train(
    data_dir='./data/augmentation/',
    save_dir='./output/baseline/',
    holefilling_kernel_size=13,
    holefilling_iter=50, 
    batch_size=8,
    epoch=1000, 
    learning_rate=1e-4,
    weight_decay=1e-1,
    alpha=1.0,
    beta=0.8,
    ## ArchBoost setting
    smooth_initial=True,
    learnable_d2n=True,
    auxiliary_loss=True,
    lambda_aux=0.1 
):     
    # save_dir setting 
    os.makedirs(save_dir, exist_ok=True)
    
    # logger setting 
    logging.basicConfig(
        filename=os.path.join(save_dir, 'log.txt'),
        filemode='w', 
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    logging.info(f"[Hyperparameters] holefilling_iter={holefilling_iter}, holefilling_kernel_size={holefilling_kernel_size}, "
                 f"batch_size={batch_size}, epoch={epoch}, learning_rate={learning_rate}, weight_decay={weight_decay}, "
                 f"alpha={alpha}, beta={beta}")

    # device setting 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"[Device] {device}")

    # define dataloader 
    dataloader = load_dataset(
        data_dir=data_dir,
        batch_size=batch_size
    )

    # define model 
    model = ArchBoost(
        holefilling_kernel_size=holefilling_kernel_size,
        holefilling_iter=holefilling_iter,
        smooth_initial=smooth_initial,
        learnable_d2n=learnable_d2n,
    )
    model = model.to(device)

    # define optimizer
    optimizer = torch.optim.AdamW(
        model.rgbd2depth.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )

    # train loop 
    for epoch in range(1,epoch+1):
        model.train()
        
        epoch_loss_sparse = 0.0
        epoch_loss_normal = 0.0
        epoch_loss_aux = 0.0
        epoch_loss_total = 0.0
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

            # loss
            if not auxiliary_loss: 
                lambda_aux=0.0
            loss_dict = acrhboost_loss(
                pred_depth=depth,
                pred_normal=normals,
                gt_sparse=sparse,
                gt_normal=normal,
                alpha=alpha,
                beta=beta,
                lambda_aux=lambda_aux,
            )

            optimizer.zero_grad()
            loss_dict['total'].backward()
            optimizer.step()

            mse_initial = F.mse_loss(initial_depth, gt, reduction='mean')
            mse_depth = F.mse_loss(depth, gt, reduction='mean')

            epoch_loss_sparse += loss_dict['sparse'].item()
            epoch_loss_normal += loss_dict['normal'].item()
            epoch_loss_aux += loss_dict['aux']
            epoch_loss_total += loss_dict['total'].item()
            epoch_mse_initial += mse_initial.item()
            epoch_mse_depth += mse_depth.item()
            num_batches += 1

        if auxiliary_loss: 
            logging.info(
                f"[Epoch {epoch}] "
                f"Loss Sparse: {epoch_loss_sparse / num_batches:.4f}, "
                f"Loss Normal: {epoch_loss_normal / num_batches:.4f}, "
                f"Loss Aux: {epoch_loss_aux / num_batches:.4f}, "
                f"Total Loss: {epoch_loss_total / num_batches:.4f}, "
                f"RMSE Initial: {math.sqrt(epoch_mse_initial / num_batches):.4f}, "
                f"RMSE Depth: {math.sqrt(epoch_mse_depth / num_batches):.4f}"
            )
        else: 
            logging.info(
                f"[Epoch {epoch}] "
                f"Loss Sparse: {epoch_loss_sparse / num_batches:.4f}, "
                f"Loss Normal: {epoch_loss_normal / num_batches:.4f}, "
                f"Total Loss: {epoch_loss_total / num_batches:.4f}, "
                f"RMSE Initial: {math.sqrt(epoch_mse_initial / num_batches):.4f}, "
                f"RMSE Depth: {math.sqrt(epoch_mse_depth / num_batches):.4f}"
            )

    # save results
    with torch.no_grad():
        # reuse last batch 
        depth_initial_np = initial_depth[0].squeeze(0).squeeze(0).cpu().numpy()    # (H, W)
        depth_pred_np = depth[0].squeeze(0).squeeze(0).cpu().numpy()    # (H, W)
        depth_gt_np = gt[0].squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
        depth_sparse_np = sparse[0].squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
        normal_pred_np = normals[0].squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        normal_gt_np = normal[0].squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        # vis results 
        save_depth_image(depth_initial_np, os.path.join(save_dir, "depth_initial.png"))
        save_depth_image(depth_pred_np, os.path.join(save_dir, "depth_pred.png"))
        save_depth_image(depth_gt_np, os.path.join(save_dir, "depth_gt.png"))
        save_depth_image(depth_sparse_np, os.path.join(save_dir, "depth_sparse.png"))
        save_normal_image(normal_pred_np, os.path.join(save_dir, "normal_pred.png"))
        save_normal_image(normal_gt_np, os.path.join(save_dir, "normal_gt.png"))
        # npy, pth 
        np.save(os.path.join(save_dir, "depth_initial.npy"), depth_initial_np)
        np.save(os.path.join(save_dir, "depth_pred.npy"), depth_pred_np)
        torch.save(model.state_dict(), os.path.join(save_dir, f"weight.pth")) 
    logging.info(f"[END] All results saved into {save_dir}")

