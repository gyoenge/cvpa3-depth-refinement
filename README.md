# Supervised Depth Refinement (SDR) 
## Computer Vision Course Coding Assignment 3

- Course: GIST Computer Vision (EC4216)
- Project Type: Supervised Depth Refinement Implementation Individual Coding Assignment

### Overview 

<p align="justify">
In this project, we implemented <b>Supervised Depth Refinement (SDR)</b> to accurately predict complete depth from the given <b>Sparse Depth, RGB Image, and Surface Normal</b>. The <b>SDR model</b> takes sparse depth and RGB as input and outputs depth and normal, where the model learns weights under the supervision of <b>Sparse Ground Truth</b> and <b>Normal Ground Truth</b> to perform depth refinement.
</p>

<p align="justify">
The <b>baseline model</b> consists of <b>HoleFiller</b>, <b>UNet</b>, and <b>Depth2Normal</b> modules, among which only <b>UNet</b> is trainable. It is trained to minimize <b>Sparse depth loss</b> and <b>Normal loss</b>. To improve the performance of the baseline model, we additionally designed and applied two boosting strategies: <b>ArchBoost (Architecture Boost)</b> and <b>DataBoost (Data-driven Boost)</b>.
</p>

<p align="justify">
<b>ArchBoost</b> enhances performance through three structural improvements: (i) <b>Smooth Hole-filling</b>, (ii) <b>Average Pooling Depth2Normal</b>, and (iii) <b>Auxiliary Depth Loss</b>. On the other hand, <b>DataBoost</b> improves performance with two data-driven approaches: (i) <b>Transfer Learning</b> for robustness, and (ii) <b>Sample Data Augmentation</b>.
</p>

<p align="center">
<img width="80%" alt="image" src="https://github.com/user-attachments/assets/5f98af5a-f961-4ec1-968d-9dd03c13e5f1" />
</p>

---

## Description

### setup

note that torch version should be matched with cuda 

```
python -m venv venv
source venv/bin/activate
pip install numpy matplotlib tqdm  # (optional) ipykernel 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128  # me: cuda12.8
```

### run 

run data augmentation 

```
python augmentation.py
```

run to train and evaluate 

```
python main.py
```

---

