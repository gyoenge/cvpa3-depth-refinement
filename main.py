from SDR.baseline.train import train as train_baseline
from SDR.baseline.inference import inference as inference_baseline
from SDR.archboost.train import train as train_archboost
from SDR.archboost.inference import inference as inference_archboost 
from SDR.test import *


"""
Baseline 
"""

# train_baseline(
#     data_dir='./data/aug0.002', #'./data/augmentation/',
#     save_dir='./output/baseline/',
#     holefilling_kernel_size=7,
#     holefilling_iter=50, 
#     batch_size=8,
#     epoch=100, 
#     learning_rate=1e-4,
#     weight_decay=1e-1,
#     alpha=1.0,
#     beta=0.7,
# )

# test_d2n(
#     data_dir='./data/ori_train/',
#     save_dir='./output/archboost/', 
#     is_learnable_d2n=False, 
# )

# inference_baseline(
#     data_dir='./data/ori_test/',
#     save_dir='./output/baseline/inference/',
#     model_path='./result_sota/baseline/weight.pth',
#     holefilling_kernel_size=7,
#     holefilling_iter=50, 
#     batch_size=1, 
# )


"""
ArchBoost
"""

# train_archboost(
#     data_dir='./data/ori_train/',
#     save_dir='./output/archboost/',
#     holefilling_kernel_size=13,
#     holefilling_iter=30, 
#     batch_size=8, 
#     epoch=10000, 
#     learning_rate=5e-3, 
#     weight_decay=1e-1,
#     lr_scheduling=True, 
#     alpha=1.0,
#     beta=0.2,
#     ## ArchBoost setting
#     smooth_initial=True,
#     learnable_d2n=True,
#     auxiliary_loss=True,
#     lambda_aux=0.2, 
#     ## Save setting
#     save_checkpoint=True, 
# )

# test_d2n(
#     data_dir='./data/ori_train/',
#     save_dir='./output/archboost/',
#     is_avgpool_d2n=True, 
# )

inference_archboost(
    data_dir='./data/ori_test/',
    save_dir='./output/archboost/inference/',
    model_path='./result_sota/archboost/weight.pth',
    holefilling_kernel_size=13,
    holefilling_iter=30, 
    batch_size=1, 
    # ArchBoost setting
    smooth_initial=True,
    learnable_d2n=True,
)
