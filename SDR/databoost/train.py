import os 
import shutil
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
    ### 
    model_path='./data/model/archboost.pth',
    has_gt=False, 
    ### 
    holefilling_kernel_size=13,
    holefilling_iter=50, 
    batch_size=8,
    epoch=1000, 
    learning_rate=1e-2,
    weight_decay=1e-1,
    lr_scheduling=True, 
    alpha=1.0,
    beta=0.2,
    ## ArchBoost setting
    smooth_initial=True,
    learnable_d2n=True,
    auxiliary_loss=True,
    lambda_aux=0.2, 
    ## Save setting
    save_checkpoint=True
):     
    if not auxiliary_loss: 
        lambda_aux=0.0

    # save_dir setting 
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
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
    logging.info(f"[Hole Filling Hyperparameters] "
                 f"kernel_size=({holefilling_kernel_size}x{holefilling_kernel_size}), iter={holefilling_iter} ")
    logging.info(f"[Train Hyperparameters] "
                 f"batch_size={batch_size}, epoch={epoch}, learning_rate={learning_rate}, weight_decay={weight_decay}, lr_scheduling={lr_scheduling}, "
                 f"alpha={alpha}, beta={beta}, lambda_aux={lambda_aux}")
    logging.info(f"[ArchBoost Ablations] "
                 f"smooth_initial={smooth_initial}, learnable_d2n={learnable_d2n}, auxiliary_loss={auxiliary_loss},")

    # device setting 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"[Device] {device}")

    # define dataloader 
    dataloader = load_dataset(
        data_dir=data_dir,
        batch_size=batch_size, 
        has_gt=has_gt,
    )

    # define model 
    model = ArchBoost(
        holefilling_kernel_size=holefilling_kernel_size,
        holefilling_iter=holefilling_iter,
        smooth_initial=smooth_initial,
        learnable_d2n=learnable_d2n,
    )
    model = model.to(device)
    # load model 
    model.load_state_dict(torch.load(model_path, map_location=device))

    # define optimizer
    optimizer = torch.optim.AdamW(
        model.rgbd2depth.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )

    # define scheduler 
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    if lr_scheduling: 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=5e-4)

    # train loop 
    sota_depth_rmse = 100000 
    # sota_loss = 100000
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
            if has_gt: 
                gt = batch['gt'].to(device)                # (B, 1, H, W)
            normal = batch['normal'].to(device)        # (B, 3, H, W) 

            initial_depth, depth, normals = model(
                rgb, # (B, 3, H, W)
                sparse,# (B, 1, H, W)
            )

            # loss
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
            
            if has_gt: 
                mse_initial = F.mse_loss(initial_depth, gt, reduction='mean')
                mse_depth = F.mse_loss(depth, gt, reduction='mean')
            else: 
                mask = (sparse > 0).float()
                mse_initial = F.mse_loss(initial_depth * mask, sparse, reduction='sum') / mask.sum()
                mse_depth = F.mse_loss(depth * mask, sparse, reduction='sum') / mask.sum()
            epoch_loss_sparse += loss_dict['sparse'].item()
            epoch_loss_normal += loss_dict['normal'].item()
            epoch_loss_aux += loss_dict['aux']
            epoch_loss_total += loss_dict['total'].item()
            epoch_mse_initial += mse_initial.item()
            epoch_mse_depth += mse_depth.item()
            num_batches += 1

        if lr_scheduling: 
            scheduler.step()
        
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

        # if lr_scheduling:         
            # logging.info(f"| lr: {scheduler.get_last_lr()[0]:.6f}")

        ## save checkpoint result 
        if save_checkpoint and (epoch%500)==0: 
            with torch.no_grad():
                depth_pred_np = depth[0].squeeze(0).squeeze(0).cpu().numpy()    # (H, W)
                normal_pred_np = normals[0].squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
                save_depth_image(depth_pred_np, os.path.join(save_dir, f"depth_pred_ep{epoch}.png"))
                save_normal_image(normal_pred_np, os.path.join(save_dir, f"normal_pred_ep{epoch}.png"))

        ## SOTA check 
        # if epoch==1:
        #     ## SOTA should predict better than initial depth 
        #     # print(f"SOTA setted: {math.sqrt(epoch_mse_initial / num_batches):.4f}")
        #     sota_depth_rmse = math.sqrt(epoch_mse_initial / num_batches)
        if (math.sqrt(epoch_mse_depth / num_batches)) < sota_depth_rmse:
            sota_depth_rmse = math.sqrt(epoch_mse_depth / num_batches)
            logging.info(
                f"[New SOTA]"
                f"Epoch {epoch}, Depth RMSE {sota_depth_rmse:.4f}"
            )
        # if epoch>400 and (epoch_loss_total / num_batches) < sota_loss: 
        #     sota_loss = epoch_loss_total / num_batches
        #     logging.info(
        #         f"[New SOTA]"
        #         f"Epoch {epoch}, Total Loss {sota_loss:.4f}"
        #     )
            ## save 
            with torch.no_grad():
                depth_pred_np = depth[0].squeeze(0).squeeze(0).cpu().numpy()    # (H, W)
                normal_pred_np = normals[0].squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
                save_depth_image(depth_pred_np, os.path.join(save_dir, f"depth_pred_SOTA.png"))
                save_normal_image(normal_pred_np, os.path.join(save_dir, f"normal_pred_SOTA.png"))
                np.save(os.path.join(save_dir, "depth_pred_SOTA.npy"), depth_pred_np)
                torch.save(model.state_dict(), os.path.join(save_dir, f"weight_SOTA.pth")) 

    # save results
    with torch.no_grad():
        # reuse last batch 
        depth_initial_np = initial_depth[0].squeeze(0).squeeze(0).cpu().numpy()    # (H, W)
        depth_pred_np = depth[0].squeeze(0).squeeze(0).cpu().numpy()    # (H, W)
        if has_gt: 
            depth_gt_np = gt[0].squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
        depth_sparse_np = sparse[0].squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
        normal_pred_np = normals[0].squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        normal_gt_np = normal[0].squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        # vis results 
        save_depth_image(depth_initial_np, os.path.join(save_dir, "depth_initial.png"))
        save_depth_image(depth_pred_np, os.path.join(save_dir, "depth_pred.png"))
        if has_gt: 
            save_depth_image(depth_gt_np, os.path.join(save_dir, "depth_gt.png"))
        save_depth_image(depth_sparse_np, os.path.join(save_dir, "depth_sparse.png"))
        save_normal_image(normal_pred_np, os.path.join(save_dir, "normal_pred.png"))
        save_normal_image(normal_gt_np, os.path.join(save_dir, "normal_gt.png"))
        # npy, pth 
        np.save(os.path.join(save_dir, "depth_initial.npy"), depth_initial_np)
        np.save(os.path.join(save_dir, "depth_pred.npy"), depth_pred_np)
        torch.save(model.state_dict(), os.path.join(save_dir, f"weight.pth")) 
    logging.info(f"[END] All results saved into {save_dir}")
    logging.info(
                f"[Final SOTA]"
                f"Depth RMSE {sota_depth_rmse:.4f}"
            )
    # logging.info(
    #         f"[Final SOTA]"
    #         f"Total Loss {sota_loss:.4f}"
    #     )
