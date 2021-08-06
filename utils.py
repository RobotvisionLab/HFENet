import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchstat import stat
from thop import profile

# Splice and save the original image, label and result image
def visualization(tensor, strIndex):
	img = tensor[0][0]
	lab = tensor[1][0]
	out = tensor[2][0]
	
	img = img.detach().cpu().squeeze().numpy()
	lab = lab.detach().cpu().squeeze().numpy()
	out = out.detach().cpu().squeeze().numpy()
	
	plt.figure()
	ax1 = plt.subplot(1,3,1)
	ax1.set_title('Input')
	plt.imshow(img[0], cmap="gray")
	ax2 = plt.subplot(1,3,2)
	ax2.set_title('Label')
	plt.imshow(lab, cmap="gray")
	ax3 = plt.subplot(1,3,3)
	ax3.set_title('Output')
	plt.imshow(out, cmap="gray")

	picName = './visualization/' + strIndex + '.jpg'
	plt.savefig(picName)
	plt.cla()
	plt.close("all")


def ImageBinarization(img, threshold=1):
	img = np.array(img)
	image = np.where(img > threshold, 1, 0)
	return image

def label_preprocess(label):
	label_pixel = ImageBinarization(label)
	return  label_pixel

def cvTotensor(img):
	img = (np.array(img[:, :, np.newaxis]))
	img = np.transpose(img,(2,0,1))
	img = (np.array(img[np.newaxis, :,:, :]))    
	tensor = torch.from_numpy(img)
	tensor = torch.as_tensor(tensor, dtype=torch.float32)
	return tensor

def cvTotensor_img(img):
	img = np.transpose(img,(2,0,1))
	img = (np.array(img[np.newaxis, :,:, :]))    
	tensor = torch.from_numpy(img)
	tensor = torch.as_tensor(tensor, dtype=torch.float32)
	return tensor

def caculate_FLOPs_and_Params(model):
	input = torch.randn(1, 3, 1408, 256)
	flops, params = profile(model, inputs=(input, ))
	print('flops: ', flops, ' params: ', params)
	return flops, params

def calFlop(model, path):
	checkpoint = torch.load(path, map_location='cpu' )
	model.load_state_dict(checkpoint['model'])
	stat(model, (3, 1408, 256))