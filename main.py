from SDR.baseline.train import train as train_baseline

train_baseline(
    data_dir='./data/aug0.002', #'./data/augmentation/',
    save_dir='./output/baseline/',
    holefilling_kernel_size=7,
    holefilling_iter=50, 
    batch_size=8,
    epoch=100, 
    learning_rate=1e-4,
    weight_decay=1e-1,
    alpha=1.0,
    beta=0.7,
)

