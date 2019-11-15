import torch, os, sys, cv2
import torch.nn as nn
from torch.nn import init
import functools
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as func
from PIL import Image

import torchvision.transforms as transforms
import numpy as np
import torch


def LoG(img):
	weight = [
		[0, 0, 1, 0, 0],
		[0, 1, 2, 1, 0],
		[1, 2, -16, 2, 1],
		[0, 1, 2, 1, 0],
		[0, 0, 1, 0, 0]
	]
	weight = np.array(weight)

	weight_np = np.zeros((1, 1, 5, 5))
	weight_np[0, 0, :, :] = weight
	weight_np = np.repeat(weight_np, img.shape[1], axis=1)
	weight_np = np.repeat(weight_np, img.shape[0], axis=0)

	weight = torch.from_numpy(weight_np).type(torch.FloatTensor).to('cuda:0')

	return func.conv2d(img, weight, padding=1)

def HFEN(output, target):
	return torch.sum(torch.pow(LoG(output) - LoG(target), 2)) / torch.sum(torch.pow(LoG(target), 2))


def l1_norm(output, target):
	return torch.sum(torch.abs(output - target)) / torch.numel(output)

def get_temporal_data(output, target):
	final_output = output.clone()
	final_target = target.clone()
	final_output.fill_(0)
	final_target.fill_(0)

	for i in range(1, 7):
		final_output[:, i, :, :, :] = output[:, i, :, :] - output[:, i-1, :, :]
		final_target[:, i, :, :, :] = target[:, i, :, :] - target[:, i-1, :, :]

	return final_output, final_target

def temporal_norm(output, target):
	return torch.sum(torch.abs(output - target)) / torch.numel(output)

def loss_func(output, temporal_output, target, temporal_target):
	ls = l1_norm(output, target)
	lg = HFEN(output, target)
	lt = temporal_norm(temporal_output, temporal_target)

	return 0.8 * ls + 0.1 * lg + 0.1 * lt, ls, lg, lt
