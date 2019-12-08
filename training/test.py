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
import imageio

root_dir = '/media/dewang/SanDisk/living_room/'
device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
m = find_max(root_dir+'RGB',4,1,1)
m = np.cumsum(m,axis=0)
preprocess(root_dir,root_dir+'RGB',root_dir+'Depth',root_dir+'Albedos',root_dir+'Normals',root_dir+'GroundTruth',m,512)
dataset = AutoEncoderData(root_dir+'RGB',root_dir+'input',root_dir+'gt',(512,512),m)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
model =   AutoEncoder(10).to(device)
size = 512
fin = np.zeros((size,size*3,3))
checkpoint = torch.load('models/living_room/autoencoder_model_elephant_45.pt')
model.load_state_dict(checkpoint['net'])
overall_step = 0
res_list = []
total_step = len(test_loader)
model.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        if i%7 !=0:
            continue
        input = data['image'].float().to(device)
        output = data['output'].float().to(device)
        for j in range(7):
            fin = np.zeros((size,size*3,3))
            input_i = input[:,j,:,:,:]
            output_i = output[:,j,:,:,:]
            fin[:,:size,:] = np.clip(input_i[0,:3,:,:].permute(1,2,0).detach().cpu().numpy(),0,1)
            fin[:,:size,:] = (fin[:,:size,:]*255).astype(np.uint8)
            pred = model(input_i,j)
            fin[:,size:size*2,:] = np.clip(pred[0].permute(1,2,0).cpu().numpy(),0,1)
            fin[:,size:size*2,:] = (fin[:,size:size*2,:]*255).astype(np.uint8)
            fin[:,size*2:,:] = np.clip(output_i[0,:,:,:].permute(1,2,0).detach().cpu().numpy(),0,1)
            fin[:,size*2:,:] = (fin[:,size*2:,:]*255).astype(np.uint8)
            res_list.append(fin)

imageio.mimsave('./network_output.gif', res_list)
