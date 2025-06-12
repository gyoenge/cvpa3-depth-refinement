import torch
# from torchvision.transforms.functional import gaussian_blur
from SDR.dataset import load_dataset
from SDR.baseline.depth2normal import Depth2Normal
from SDR.archboost.avgpool_d2n import LearnableD2N
from SDR.utils import *


def test_d2n(
    data_dir='./data/ori_train/',
    save_dir='./output/baseline/',
    is_learnable_d2n=False, 
): 
    batch_size=1

    # save_dir setting 
    os.makedirs(save_dir, exist_ok=True)

    # device setting 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define dataloader 
    dataloader = load_dataset(
        data_dir=data_dir,
        batch_size=batch_size
    )

    # define model 
    if is_learnable_d2n: 
        d2n_model = LearnableD2N()
    else: 
        d2n_model = Depth2Normal()
    d2n_model = d2n_model.to(device)

    # d2n infer
    with torch.no_grad():
        d2n_model.eval()
        for batch in dataloader:
            # rgb = batch['rgb'].to(device)              # (B, 3, H, W)
            # sparse = batch['sparse_depth'].to(device)  # (B, 1, H, W)
            gt = batch['gt'].to(device)                # (B, 1, H, W)
            normal = batch['normal'].to(device)        # (B, 3, H, W) 

            # gt_blurred = gaussian_blur(gt, kernel_size=5, sigma=1.0)

            normals = d2n_model(
                gt # (B, 1, H, W)
                # gt_blurred
            ) # (B, 3, H, W)

            # only use first batch 
            break
        
        # only use first sample 
        normal_pred_np = normals[0].squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        save_normal_image(normal_pred_np, os.path.join(save_dir, "test_gt_d2n.png"))
        normal_gt_np = normal[0].squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        save_normal_image(normal_gt_np, os.path.join(save_dir, "test_gt_normal.png"))

