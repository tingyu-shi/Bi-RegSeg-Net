# Bi-RegSeg-Net

## config environment
```code
conda create -n sigma python=3.10.*
conda activate sigma
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install tqdm opencv-python opencv-python-headless torch-summary pywavelets timm fvcore==0.1.5.post20221221 wandb kornia
cd ./nets/VMamba/selective_scan && pip install .
```