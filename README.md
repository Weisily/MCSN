# MCSN
This repository is an official PyTorch implementation of the paper 
"A lightweight multi-scale channel attention network for image super-resolution".
The code is built on EDSR (Torch) and tested on Ubuntu 14.04 environment  with TitanX 2080Ti GPU.
Dependencies
•	Python 3.6
•	PyTorch = 1.2.0
•	numpy
•	skimage
•	imageio
•	matplotlib
•	tqdm
Data
all scale factor(x2,x3,x4,x8) data:
training data DIV2K(800 training + 100 validtion images)
benchmark data (Set5, Set14, B100, Urban100, Manga109)
Train
1, Cd to './MCSN/src', run the following commands to train models.
Test
1,Cd to './code/src', run the following commands to get result with paper reported.
