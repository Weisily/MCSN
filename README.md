# MCSN
This repository is an official PyTorch implementation of the paper 
"A lightweight multi-scale channel attention network for image super-resolution".
The code is built on EDSR (Torch) and tested on Ubuntu 14.04 environment  with TitanX 2080Ti GPU.
# Dependencies
*	Python 3.6
*	PyTorch = 1.2.0
*	numpy
* skimage
* imageio
* matplotlib
* tqdm
# Data
All scale factor(x2,x3,x4,x8) data:
1. Training data DIV2K(800 training + 100 validtion images)
2. Benchmark data (Set5, Set14, B100, Urban100, Manga109)
# Train
1.Cd to './MCSN/src', run the following commands to train models.
```python
python main.py --model MCSN--scale 2 --save mcsn_x2  --n_resblocks 3  --lr 1e-4  --n_feats 64 --res_scale 1 --batch_size 16 --n_threads 6 
python main.py --model MCSN--scale 3 --save mcsn_x3  --n_resblocks 3  --lr 1e-4  --n_feats 64 --res_scale 1 --batch_size 16 --n_threads 6 
python main.py --model MCSN--scale 4 --save mcsn_x4  --n_resblocks 3  --lr 1e-4  --n_feats 64 --res_scale 1 --batch_size 16 --n_threads 6 
python main.py --model MCSN--scale 8 --save mcsn_x8  --n_resblocks 3  --lr 1e-4  --n_feats 64 --res_scale 1 --batch_size 16 --n_threads 6 
```
# Test
```python
python main.py --model MCSN --data_test Set5+Set14+B100+Urban100+Manga109  --scale 4 --pre_train ../experiment/mscn_x4/model/model_best.pt --test_only  --self_ensemble
```
# Results
![Network](https://github.com/Weisily/MCSN/Figs/Network.png)
![Parameters](https://github.com/Weisily/MCSN/Figs/Parameters.png)
![CSAM](https://github.com/Weisily/MCSN/Figs/CSAM.png)
![PSNR&SSIMX2](https://github.com/Weisily/MCSN/Figs/PSNR&SSIMX2.png)
