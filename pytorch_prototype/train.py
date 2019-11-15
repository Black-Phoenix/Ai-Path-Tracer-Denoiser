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
import torch, argparse, pdb

from recurrent_autoencoder_model import *
from dataloader import *
from loss import *
from tensorboard import *

logger = Logger('./logs')
device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
m = find_max('../train_data/scenes',4)
dataset = AutoEncoderData('../train_data','../train_data/scenes','../train_data/scenes','../train_data/scenes',(256,256), m)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
model =   AutoEncoder(7).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
overall_step = 0
total_step = len(train_loader)
for epoch in range(20):
    total_loss = 0
    total_loss_num = 0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        input = data['image'].to(device)
        label = data['output'].to(device)
        outputs = torch.zeros_like(label)
        targets = torch.zeros_like(label)
        loss_final = 0
        ls_final = 0
        lg_final = 0
        lt_final = 0
        for j in range(7):
            input_i = input[:,j,:,:]
            label_i = label[:,j,:,:]
            model.set_input(input_i)
            if j == 0:
                model.reset_hidden()
            output = model()
            outputs[:,j,:,:] = output
            targets[:,j,:,:] = label_i
        temporal_output, temporal_target = get_temporal_data(outputs, targets)

        for j in range(7):
            output = outputs[:,j,:,:]
            target = targets[:,j,:,:]
            t_output = temporal_output[:,j,:,:]
            t_target = temporal_target[:,j,:,:]
            l, ls, lg, lt = loss_func(output, t_output, target, t_target)
            loss_final += l
            ls_final += ls
            lg_final += lg
            lt_final += lt
        info = {("Total"):loss_final.item(),("L1"):ls_final.item(), ("HFEN"):lg_final.item(),("Temporal"):lt_final.item()}
        if i%50 == 0:
             print ('Epoch [{}/{}], Step [{}/{}], Total Loss: {:.4f}, L1 Loss: {:.4f}, HFEN Loss: {:.4f}, Temporal Loss: {:.4f}'.format(epoch+1,
                                                                                                    20, i+1, total_step, loss_final.item(),ls_final.item(),lg_final.item(),lt_final.item()))
        for tag, value in info.items():
            logger.scalar_summary(tag, value, overall_step+1)
        overall_step += 1
        loss_final.backward(retain_graph=False)
        optimizer.step()

        total_loss += loss_final.item()
        total_loss_num += 1
    print("Average loss over Epoch {} = {}".format(epoch, total_loss/total_loss_num))
