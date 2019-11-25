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

root_dir = '../Train/'
# logger = Logger('./logs')
device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
m = find_max(root_dir+'RGB',1,2)
inputs, outputs = preprocess(root_dir,root_dir+'RGB',root_dir+'Depth',root_dir+'Albedos',root_dir+'Normals',root_dir+'GroundTruth',m)
dataset = AutoEncoderData(root_dir+'RGB',inputs,outputs,(256,256),m)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
model =   AutoEncoder(10).to(device)
fin = np.zeros((256,768,3))
checkpoint = torch.load('autoencoder_model_1.pt')
model.load_state_dict(checkpoint['net'])
overall_step = 0
res_list = []
total_step = len(test_loader)
with torch.no_grad():
    for i, data in enumerate(test_loader):
        if i%7 !=0:
            continue
        input = data['image'].float().to(device)
        output = data['output'].float().to(device)
        for j in range(7):
            fin = np.zeros((256,768,3))
            # fig, ax = plt.subplots(3)
            input_i = input[:,j,:,:,:]
            output_i = output[:,j,:,:,:]
            fin[:,:256,:] = input_i[0,:3,:,:].permute(1,2,0).detach().cpu().numpy()
            # ax[0].imshow(fin[:,:256,:])
            # ax[0].set_title("Noisy Image")
            model.set_input(input_i)
            if j == 0:
                model.reset_hidden()
            pred = model()
            fin[:,256:512,:]  = pred[0].permute(1,2,0).cpu().numpy()
            # ax[1].imshow(fin[:,256:512,:])
            # ax[1].set_title("Denoised Image")
            fin[:,512:,:] = output_i[0,:,:,:].permute(1,2,0).detach().cpu().numpy()
            # ax[2].imshow(fin[:,512:,:])
            # ax[2].set_title("Ground Truth")
            # plt.show()
            res_list.append(fin)
imageio.mimsave('./eval_testimg.gif', res_list)
