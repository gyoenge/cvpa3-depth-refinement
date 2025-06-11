from SDR.baseline.train import train as train_baseline
from SDR.archboost.train import train as train_archboost

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

train_archboost(
    data_dir='./data/ori_train/',
    save_dir='./output/archboost/',
    holefilling_kernel_size=13,
    holefilling_iter=50, 
    batch_size=8, 
    epoch=1000, 
    learning_rate=1e-2,
    weight_decay=1e-1,
    alpha=1.0,
    beta=0.8,
    ## ArchBoost setting
    smooth_initial=True,
    learnable_d2n=True,
    auxiliary_loss=False,
    # auxiliary_loss=True,
    # lambda_aux=0.1 
)
