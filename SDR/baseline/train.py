import os 
import logging
import math
import torch 
import torch.nn.functional as F
from SDR.dataset import load_dataset
from SDR.baseline.model import Baseline

def train(
    data_dir='./data/augmentation/',
    save_dir='./output/baseline/',
    holefilling_kernel_size=(7,7),
    holefilling_iter=50, 
    batch_size=8,
    epoch=1000,
    learning_rate=1e-4,
    weight_decay=1e-1,
    alpha=1.0,
    beta=0.7,
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
                 f"alpha={alpha}, beta={beta}, ")

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

    # define optimizer
    optimizer = torch.optim.AdamW(
        model.rgbd2depth.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )

    # train loop 
    for epoch in range(1,epoch+1):
        model.train()
        
        epoch_loss_initial = 0.0
        epoch_loss_sparse = 0.0
        epoch_loss_normal = 0.0
        epoch_loss_total = 0.0
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

            mask_sparse = (sparse > 0).float()  # (B, 1, H, W)
            loss_initial = F.mse_loss(initial_depth * mask_sparse, sparse, reduction='sum') / (mask_sparse.sum() + 1e-8)
            loss_sparse = F.mse_loss(depth * mask_sparse, sparse, reduction='sum') / (mask_sparse.sum() + 1e-8)  # or F.l1_loss(..)
            loss_normal = F.l1_loss(normals, normal)  # or F.mse_loss(..)
            loss = alpha * loss_sparse + beta * loss_normal

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_initial += loss_initial.item()
            epoch_loss_sparse += loss_sparse.item()
            epoch_loss_normal += loss_normal.item()
            epoch_loss_total += loss.item()
            num_batches += 1

        logging.info(
            f"[Epoch {epoch}] "
            f"Loss Sparse: {epoch_loss_sparse / num_batches:.4f}, "
            f"Loss Normal: {epoch_loss_normal / num_batches:.4f}, "
            f"Total Loss: {epoch_loss_total / num_batches:.4f}"
        )
        logging.info(
            f"RMSE Initial: {math.sqrt(epoch_loss_initial / num_batches):.4f}, "
            f"RMSE Sparse: {math.sqrt(epoch_loss_sparse / num_batches):.4f}"
        )

    # save results
    # (TBD)
