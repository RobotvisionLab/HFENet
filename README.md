# HFENet
A Lightweight Hand-crafted Feature Enhanced CNN for Ceramic Tile Surface Defects Detection

# Main structure
![image](https://github.com/RobotvisionLab/HFENet/blob/main/image/HFE_mainstructure.tif)

# Abstract
Inkjet printing technology can make tiles with very rich and realistic patterns, so it is widely adopted in the ceramic industry. However, the frequent nozzle blockage and inconsistent ink jet volume by inkjet printing devices, usually leads to defects such as stayguy and color blocks in the tile surface. Especially, the stayguy in complex pattern are difficult to identify by naked eyes due to its low resolution, bringing great challenge to tile quality inspection. Nowadays, the machine learning is employed to address the issues. The existing machine learning methods based on hand-crafted features are capable of stayguy detection of the tiles with simple pattern, but not applicable for complex patterns due to poor generalization performance. The emerging deep learning based methods have the potential to be applied for the complex patterns, but cannot achieve real-time detection due to high complexity. In this paper, a lightweight hand-crafted feature enhanced convolutional neural network (named HFENet) is proposed for rapid defect detection of tile surface. Firstly, we expand the original image from single channel to three channels by global histogram equalization and image channel overlaying. Secondly, for the special shape of stayguy which is usually vertical, we embed the extended vertical edge detection operator (Prewitt) as convolution kernel into HFENet to extract the hand-crafted vertical edge features of the test image and eliminate the interference of complex pattern on the feature extraction. Thirdly, the 5x1 asymmetric convolution kernel with dilation rate of 2 is used to improve the utilization of convolution kernel and reduce the complexity of the model. The experiments performed on the image dataset captured by high-resolution industrial cameras indicates that the HFENet outperformed in the comparison with the state-of-the-art semantic segmentation network DeepLabV3+, with less FLOPs by 15 times and faster by 4 times when improving the detection accuracy by 14%.

# Environment for train and test
#### python = 3.6.0
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
