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
from preprocess import preprocess

def find_max(dir, num_scenes):
    m = np.zeros(num_scenes+1)
    for file in os.listdir(dir):
        splits = file.split('_')
        m[int(splits[0])] = max(m[int(splits[0])], int(splits[1][:-4]))
    return m

class AutoEncoderData(Dataset):

    def __init__(self, scenes_dir, inputs, outputs, size, m):
        super(AutoEncoderData, self).__init__()

        self.images = sorted(os.listdir(scenes_dir))
        self.inputs = inputs
        self.outputs = outputs
        self.width = size[0]
        self.height = size[1]
        self.m = m

    def __getitem__(self, index):
        input = torch.zeros(7,self.height, self.width,10)
        output = torch.zeros(7,self.height, self.width,3)
        image_name = self.images[index]
        splits = image_name.split('_')
        start = index
        if index > int(self.m[int(splits[0])] - 7):
            start = int(self.m[int(splits[0])] - 7)
        input = torch.from_numpy(self.inputs[start:start+7,:,:,:].astype(np.float))
        output = torch.from_numpy(self.outputs[start:start+7,:,:,:].astype(np.float))

        return {'image':input.permute(0,3,1,2),'output':output.permute(0,3,1,2)}

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    m = find_max('../test/RGB',10)
    print(m)
    inputs, outputs = preprocess('../test','../test/RGB','../test/Depth','../test/Albedos','../test/Normals',m)
    dataset = AutoEncoderData('../test/RGB',inputs,outputs,(256,256),m)
    data_num = 1
    data = dataset[data_num]
    print(data['image'].shape)
    print(data['output'].shape)
