# Bi-RegSeg-Net

## config environment
```code
conda create -n sigma python=3.10.*
conda activate sigma
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.24.4 tqdm opencv-python opencv-python-headless torch-summary pywavelets timm fvcore==0.1.5.post20221221 kornia==0.8.0 ninja==1.11.1.4 einops==0.8.1 wandb
cd ./nets/VMamba/selective_scan && pip install .
```

torchrun --nproc-per-node=2 train_RegSeg_distributed.py