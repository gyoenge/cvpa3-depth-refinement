# cvpa3-depth-refinement
computer vision course programming assignment 3

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

