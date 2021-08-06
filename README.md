# HFENet
A Lightweight Hand-crafted Feature Enhanced CNN for Ceramic Tile Surface Defects Detection

# Main structure
![image](https://github.com/RobotvisionLab/HFENet/blob/main/image/HFE_mainstructure.tif)

# Abstract
A lightweight hand-crafted feature enhanced CNN(HFENet) is proposed for ceramic tile surface defects detection. Firstly, we expand the original image from single channel to three channels by global histogram equalization and image channel adding. Secondly, for the special shape of stayguy which is usually vertical, we embed the extended vertical edge detection operator (Prewitt) as convolution kernel into HFENet to extract the hand-crafted vertical edge features of the test image and eliminate the interference of complex pattern on the feature extraction. Thirdly, the 5x1 asymmetric convolution kernel with dilation rate of 2 is used to improve the utilization of convolution kernel and reduce the complexity of the model. Experiments on tile dataset captured by high-resolution industrial cameras demonstrate the superior performance of HFENet with 15× less FLOPs and 4× faster than the existing state-of-the-art semantic segmentation networks and lightweight networks while providing comparable accuracy.

# Environment for train and test
#### torch = 1.5.0
#### torchstat = 0.0.7
#### torchvision = 0.6.0
#### thop = 0.0.31
#### opencv-python = 4.2.0

# Dataset
[Tile surface images with ten different patterns taken by a high-definition industrial camera](https://drive.google.com/drive/folders/1n2u-sAk_DXCr9bd_USaVTw_J8WSj6iXJ?usp=sharing)

# How to train and test
###  1.train
```
python main.py --train
```
###  2.test
```
python main.py --test
```
###  3.caculate the parameters, FLOPs and MAC of the net
```
python main.py --info
```

###  4.run a demo
```
python main.py --demo
```
