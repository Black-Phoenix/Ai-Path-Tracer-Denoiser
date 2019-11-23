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
import torch, argparse, pdb

from recurrent_autoencoder_model import *
from dataloader import *
from loss import *
from tensorboard import *

logger = Logger('./logs')
device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
m = find_max('../test_data/scenes',4)
dataset = AutoEncoderData('../test_data','../test_data/scenes','../test_data/scenes','../test_data/scenes',(256,256), m)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
model =   AutoEncoder(10).to(device)
checkpoint = torch.load('autoencoder_model.pt')
model.load_state_dict(checkpoint['net'])
overall_step = 0
total_step = len(test_loader)
with torch.no_grad():
    for i, data in enumerate(test_loader):
        input = data['image'].to(device)
        for j in range(7):
            fig, ax = plt.subplots(2)
            input_i = input[:,j,:,:,:]
            ax[0].imshow(input_i[0,:3,:,:].permute(1,2,0).detach().cpu().numpy())
            ax[0].set_title("Noisy Image")
            model.set_input(input_i)
            if j == 0:
                model.reset_hidden()
            output = model()
            ax[1].imshow(output[0].permute(1,2,0).cpu().numpy())
            ax[1].set_title("Denoised Image")
            plt.show()
