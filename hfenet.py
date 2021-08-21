########Architecture of our proposed model (HFENet)#########

import torch
import torch.nn as nn
import numpy as np

#convolution Layer
class convLayer(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
		super(convLayer, self).__init__()

		self.activate = nn.ReLU(inplace=True)
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
			padding=padding, dilation=dilation, bias=False)
		
		self.norm = nn.InstanceNorm2d(num_features=out_channels, affine=True, track_running_stats=False)

	def forward(self, x):
		x = self.conv(x)
		x = self.norm(x)
		x = self.activate(x)
		return x

#Residual ModuleA
class ResModule_A(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ResModule_A, self).__init__()

		self.activate = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2, dilation=1, bias=False)
		self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5,1), stride=1, padding=(4,0), dilation=(2,1) , bias=False)
		self.norm1 = nn.InstanceNorm2d(num_features=out_channels, affine=True, track_running_stats=False)
		self.norm2 = nn.InstanceNorm2d(num_features=out_channels, affine=True, track_running_stats=False)

	def forward(self, x):
		shortcut = self.conv1(x)
		shortcut = self.norm1(shortcut)
		shortcut = self.activate(shortcut)
		shortcut = self.conv2(shortcut)
		shortcut = self.norm2(shortcut)
		return self.activate(x+shortcut)

#Residual ModuleB
class ResModule_B(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ResModule_B, self).__init__()

		self.activate = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
		self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5,1), stride=1, padding=(4,0), dilation=(2,1) , bias=False)
		self.norm1 = nn.InstanceNorm2d(num_features=out_channels, affine=True, track_running_stats=False)
		self.norm2 = nn.InstanceNorm2d(num_features=out_channels, affine=True, track_running_stats=False)

	def forward(self, x):
		
		shortcut = self.conv1(x)
		shortcut = self.norm1(shortcut)
		shortcut = self.activate(shortcut)
		shortcut = self.conv2(shortcut)
		shortcut = self.norm2(shortcut)
		return self.activate(x+shortcut)


#Feature Fusion Component
class FeatureFusion(nn.Module):
	def __init__(self):
		super(FeatureFusion, self).__init__()

		self.Prewitt_Operator = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2, dilation=1, groups=3, bias=False)

	def forward(self, x):
		vertical_edge = self.Prewitt_Operator(x)
		x = torch.cat([x, vertical_edge], dim=1)
		return x

#HFENet
class HFENet(nn.Module):
	def __init__(self):
		super(HFENet, self).__init__()

		#feature fusion component
		self.featurefusion  = FeatureFusion()

		#first single convolution layer 5x5
		self.firstConvLayer = convLayer(in_channels=6, out_channels=32, kernel_size=5, stride=1, padding=2, dilation=1)

		#residual moduleA
		self.resmoduleA = ResModule_A(in_channels=32, out_channels=32)

		#six residual moduleB
		self.resmoduleB1 = ResModule_B(in_channels=32, out_channels=32)
		self.resmoduleB2 = ResModule_B(in_channels=32, out_channels=32)
		self.resmoduleB3 = ResModule_B(in_channels=32, out_channels=32)
		self.resmoduleB4 = ResModule_B(in_channels=32, out_channels=32)
		self.resmoduleB5 = ResModule_B(in_channels=32, out_channels=32)
		self.resmoduleB6 = ResModule_B(in_channels=32, out_channels=32)

		#three upsample layers
		self.upsample1_conv = convLayer(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1)
		self.upsample2_conv = convLayer(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1)
		self.upsample3_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
		self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

		#maxpool layer
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

		#activation function of upsample3_conv
		self.sigmoid = nn.Sigmoid()

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight.data)
			elif isinstance(m, nn.InstanceNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

		#prewitt operator
		prewittWeight = np.array([
				   [2, 1, 0, -1, -2],
				   [2, 1, 0, -1, -2],
				   [2, 1, 0, -1, -2],
				   [2, 1, 0, -1, -2],
				   [2, 1, 0, -1, -2]], np.float32)

		#embed the operator as convolution kernel
		print('shape of old weight : ', self.featurefusion.Prewitt_Operator.weight.shape)
		prewittWeight = prewittWeight[np.newaxis, :]
		prewittWeight = np.expand_dims(prewittWeight,0).repeat(3, axis=0)
		print('shape of my weight: ', prewittWeight.shape)
							
		prewittWeight = torch.FloatTensor(prewittWeight)
		self.featurefusion.Prewitt_Operator.weight = torch.nn.Parameter(data=prewittWeight, requires_grad=False)
		print('Operator embedding successful!')
		print('shape of new weight: ', self.featurefusion.Prewitt_Operator.weight.shape)
		print(self.featurefusion.Prewitt_Operator.weight)

	def forward(self, x):
		x = self.featurefusion(x)
		x = self.firstConvLayer(x)
		x_bk1 = x

		x = self.pool(x)
		x = self.resmoduleA(x)
		x_bk2 = x

		x = self.pool(x)
		x = self.resmoduleB1(x)
		x = self.resmoduleB2(x)
		x = self.resmoduleB3(x)
		x_bk3 = x

		x = self.pool(x)
		x = self.resmoduleB4(x)
		x = self.resmoduleB5(x)
		x = self.resmoduleB6(x)

		x = self.upsample(x)
		x = torch.cat([x, x_bk3], dim=1)
		x = self.upsample1_conv(x)
		x = self.upsample(x)
		x = torch.cat([x, x_bk2], dim=1)
		x = self.upsample2_conv(x)
		x = self.upsample(x)
		x = torch.cat([x, x_bk1], dim=1)
		x = self.upsample3_conv(x)
		x = self.sigmoid(x)

		return x


if __name__ == '__main__':
	hfenet = HFENet()
	print(hfenet)

