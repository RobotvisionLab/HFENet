from hfenet import HFENet
from evaluation import Bin_classification_cal
import utils
import logging
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import cv2
import time
import sys


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train or test the model.')

    parser.add_argument(
        "--train",
        action="store_true",
        help="Define if we wanna to train the net"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Define if we wanna to test the net"
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Define if we wanna to caculate the FLOPs and Params of the net"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Define if we wanna to run a demo"
    )
    return parser.parse_args()

def get_logger(log_path='log_path'):
	if not os.path.exists(log_path):
		os.mkdir(log_path)
	timer = time.strftime("%Y-%m-%d-%H-%M-%S_", time.localtime())
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter('[%(levelname)s]   %(asctime)s    %(message)s')
	txthandle = logging.FileHandler((log_path+'/'+timer+'log.txt'))
	txthandle.setFormatter(formatter)
	logger.addHandler(txthandle)
	return logger

# caculate the evaluation metric
def caculate(output, label, clear=False):
	cal = Bin_classification_cal(output, label, 0.5, clear)
	return cal.caculate_total()

def del_models(file_path, count=5):
	dir_list = os.listdir(file_path)
	if not dir_list:
		print('file_path is empty: ', file_path)
		return
	else:
		dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
		print('dir_list: ', dir_list)
		if len(dir_list) > 5:
			os.remove(file_path + '/' + dir_list[0])

		return dir_list

def getInput_and_Label_generator(data_path):
	img_Path = data_path + "/img"
	l = os.listdir(img_Path)
	random.shuffle(l)
	for filename in l:
		img_name = img_Path + '/' + filename
		label_name = data_path + '/lab/' + filename.split('.')[0] + "_label.bmp"
		img = cv2.imread(img_name, 0)
		img_filters  = cv2.equalizeHist(img)
		img_add = cv2.add(img, img_filters)
		img_merge = cv2.merge([img, img_filters, img_add])

		lab = cv2.imread(label_name, 0)
		img = utils.cvTotensor_img(img_merge)
		lab = utils.cvTotensor(utils.label_preprocess(lab))

		yield img, lab
		
def getInput_and_Label_generator_valid(data_path):
	img_Path = data_path + "/img"
	l = os.listdir(img_Path)

	for filename in l:
		img_name = img_Path + '/' + filename
		label_name = data_path + '/lab/' + filename.split('.')[0] + "_label.bmp"
		img = cv2.imread(img_name, 0)
		img_filters  = cv2.equalizeHist(img)
		img_add = cv2.add(img, img_filters)
		img_merge = cv2.merge([img, img_filters, img_add])

		lab = cv2.imread(label_name, 0)
		img = utils.cvTotensor_img(img_merge)
		lab = utils.cvTotensor(utils.label_preprocess(lab))

		yield img, lab

iterations = 0
net = HFENet()
net.cuda()

criterion = nn.BCELoss(weight=None, reduction='mean')
optimizer = optim.Adam(net.parameters(), lr = 0.01)
valid_path = "./HFENet_Dataset/valid"
positive_path = "./HFENet_Dataset/train/defective"
negative_path = "./HFENet_Dataset/train/no_defective"
model_path = "./checkpoint"
log_path = "./log"


def train(net, epoch, iterations, loss_stop, positive_path, negative_path):
	net.train()
	epoch_loss = 0.0
	print('train...')
	g_postive = getInput_and_Label_generator(positive_path)
	g_negative = getInput_and_Label_generator(negative_path)

	for iters in tqdm(range(iterations)):
		for index in range(2):
			if index == 0:
				inputs, labels = next(g_postive)
			else:
				inputs, labels = next(g_negative)

			inputs = inputs.cuda()
			labels = labels.cuda()

			optimizer.zero_grad()
			outputs = net(inputs)

			lab = labels.detach().cpu().squeeze().numpy()
			out = outputs.detach().cpu().squeeze().numpy()

			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			epoch_loss += loss

	epoch_loss_mean = epoch_loss / iterations
	print('Train Epoch: {}\t Total Loss: {:.6f}\t Average Loss: {:.6f}'.format(epoch, epoch_loss.item(), epoch_loss_mean.item()))
	logger.info('Train Epoch:[{}] , loss: {:.6f}'.format(epoch, epoch_loss.item()))
	if epoch_loss < loss_stop:
		return True, epoch_loss
	else:
		return False, epoch_loss

def valid(net, epoch, img_path):
	#net.eval()
	valid_loss = 0.0
	img_Path = img_path + "/img"
	l = os.listdir(img_Path)
	iterations = len(l)
	print('img_Path: ', img_Path, 'len: ', iterations)
	g_data = getInput_and_Label_generator_valid(img_path)
	IoU1 = 0
	IoU2 = 0
	MIoU = 0
	PA = 0

	total_time = 0
	with torch.no_grad():
		# for iters in tqdm(range(iterations)):
		for iters in range(iterations):
			inputs, labels = next(g_data)

			inputs = inputs.cuda()
			labels = labels.cuda()

			optimizer.zero_grad()

			torch.cuda.synchronize()
			begin_time = time.perf_counter()

			outputs = net(inputs)

			torch.cuda.synchronize()
			end_time = time.perf_counter()

			interval_time = end_time - begin_time
			total_time += interval_time
			print('detect time:', interval_time,'s')
			lab = labels.detach().cpu().squeeze().numpy()
			out = outputs.detach().cpu().squeeze().numpy()

			PA, FP, FN = caculate(out, lab, (not bool(iters)))

			strIndex = str(epoch) + '_valid_' + str(iters)
			utils.visualization([inputs, labels, outputs], strIndex)

			valid_loss += criterion(outputs, labels)

		print('average detect time:', total_time/iterations,'s')

		valid_loss_mean = valid_loss / iterations
		print('           Valid Epoch: {}\t Total Loss: {:.6f}\t Average Loss: {:.6f}\t FP: {}\t FN: {}\t FN+FP: {}\t PA: {:.6f}'.format(epoch, valid_loss.item(), valid_loss_mean.item(), FP, FN, FN+FP, PA))
		logger.info('         Valid Epoch:[{}] , loss: {:.6f}, FP: {}\t FN: {}\t FN+FP: {}\t PA: {:.6f}'.format(epoch, valid_loss.item(), FP, FN, FN+FP, PA))

def main(mode, epochs = 100):
	print(net)

	img_Path = positive_path + "/img"
	l = os.listdir(img_Path)
	iterations = len(l)
	# print('img_Path: ', img_Path, 'iterations: ', iterations)

	if os.path.exists(model_path):
		dir_list = os.listdir(model_path)
		if len(dir_list) > 0:
			dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(model_path, x)))
			print('dir_list: ', dir_list)
			last_model_name = model_path + '/' + dir_list[-1]

			if mode == 0:
				utils.calFlop(model=HFENet(), path=last_model_name)
				return

			checkpoint = torch.load(last_model_name)
			net.load_state_dict(checkpoint['model'])
			# params = net.state_dict().keys()
			# for i, j in enumerate(params):
			# 	print(i, j)
			last_epoch = checkpoint['epoch']
			loss = checkpoint['loss']
			print('load epoch {} succeed! loss: {:.6f} '.format(last_epoch, loss))
		else:
			last_epoch = 0
			print('no saved model')
	else:
		last_epoch = 0
		print('no saved model')

	if last_epoch == 0 and mode == -1:
		return

	for epoch in range(last_epoch+1, epochs+1):
		if mode == 1:
			ret, loss = train(net=net, epoch=epoch, iterations=iterations, loss_stop=0.01, positive_path=positive_path, negative_path=negative_path)
			state = {'model':net.state_dict(),'epoch':epoch, 'loss':loss}
			model_name = model_path + '/model_epoch_' + str(epoch) + '.pth'
			torch.save(state, model_name)
		else:
			valid(net, epoch, valid_path)
			ret = True

		# del_models(model_path)
		if ret:
		  break

	print('Done.')


def demo():
	if os.path.exists(model_path):
		dir_list = os.listdir(model_path)

		if len(dir_list) > 0:
			dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(model_path, x)))
			print('dir_list: ', dir_list)
			last_model_name = model_path + '/' + dir_list[-1]
			checkpoint = torch.load(last_model_name)
			net.load_state_dict(checkpoint['model'])
			last_epoch = checkpoint['epoch']
			loss = checkpoint['loss']
			print('load epoch {} succeed! loss: {:.6f} '.format(last_epoch, loss))
			print('start test...')
			print('load img...')
			img = cv2.imread('test.bmp', 0)
			print('img preprocessing...')
			img_filters  = cv2.equalizeHist(img)
			img_add = cv2.add(img, img_filters)
			img_merge = cv2.merge([img, img_filters, img_add])
			img = utils.cvTotensor_img(img_merge).cuda()
			print(img.shape)
			print('inferencing...')
			outputs = net(img)
			print('inference done.')
			out = outputs.detach().cpu().squeeze().numpy()*255
			print('save to result.bmp')
			cv2.imwrite('result.bmp', out)
		else:
			print('no model!')

		sys.exit(0)

if __name__ == '__main__':
	args = parse_arguments()
	mode = 0

	if args.demo:
		demo()

	if args.train:
		mode = 1
	elif args.test:
		mode = -1
	elif args.info:
		mode = 0

	logger = get_logger(log_path)
	print('mode: ', mode)
	main(mode=mode)




