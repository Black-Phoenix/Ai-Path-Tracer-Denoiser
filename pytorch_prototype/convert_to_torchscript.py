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
# from tensorboard import *
import imageio

root_dir = '../train_2/'
device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
m = find_max(root_dir+'RGB',15,2, 5)
dataset = AutoEncoderData(root_dir+'RGB',inputs,outputs,(512,512),m)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
data = next(iter(test_loader))
model =  AutoEncoder(10).to(device)
checkpoint = torch.load('model/autoencoder_model_4_2_168.pt')
model.load_state_dict(checkpoint['net'])
traced_script_module = torch.jit.trace(model.forward, data['image'][:,0,:,:,:].float().to(device))
traced_script_module.save("cpp_autoencoder_4_2_168.pt")
