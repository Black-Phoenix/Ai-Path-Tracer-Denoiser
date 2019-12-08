import torch, os, sys, cv2
import torch.nn as nn
from torch.nn import init
import functools
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as func
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import torch
import math

def LoG(img):
	weight = [
        [0,1,0],
        [1,-4,1],
        [0,1,0]
	]
	weight = np.array(weight)

	weight_np = np.zeros((1, 1, 3, 3))
	weight_np[0, 0, :, :] = weight
	weight_np = np.repeat(weight_np, img.shape[1], axis=1)
	weight_np = np.repeat(weight_np, img.shape[0], axis=0)

	weight = torch.from_numpy(weight_np).type(torch.FloatTensor).to('cuda:0')

	return func.conv2d(img, weight, padding=1)

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter.cuda()


def HFEN(output, target):
	filter = get_gaussian_kernel(5,1.5,3)
	gradient_p = filter(target)
	gradient_o = filter(output)
	gradient_p = LoG(gradient_p)
	if torch.max(gradient_p) != 0:
		gradient_p = gradient_p/torch.max(gradient_p)
	gradient_o = LoG(gradient_o)
	if torch.max(gradient_o) !=0:
		gradient_o = gradient_o/torch.max(gradient_o)
	criterion = nn.L1Loss()
	return criterion(gradient_p, gradient_o)


def l1_norm(output, target):
	criterion = nn.L1Loss()
	return criterion(target,output)

def get_temporal_data(output, target):
	final_output = torch.zeros_like(output)
	final_target = torch.zeros_like(target)
	for i in range(1, 7):
		final_output[:, i, :, :, :] = (output[:, i, :, :] - output[:, i-1, :, :])
		final_target[:, i, :, :, :] = (target[:, i, :, :] - target[:, i-1, :, :])

	return final_output, final_target

def temporal_norm(output, target):
	criterion = nn.L1Loss()
	return criterion(target,output)

def loss_func(output, temporal_output, target, temporal_target):
	ls = l1_norm(output, target)
	lg = HFEN(output, target)
	lt = temporal_norm(temporal_output, temporal_target)

	return ls, lg, lt
