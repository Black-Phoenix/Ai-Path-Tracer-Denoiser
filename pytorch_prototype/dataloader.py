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

def find_max(dir, num_scenes):
    m = np.zeros(num_scenes+1)
    for file in os.listdir(dir):
        splits = file.split('_')
        m[int(splits[1])] = max(m[int(splits[1])], int(splits[2][:-7]))
    return m
class AutoEncoderData(Dataset):

    def __init__(self, dir, scenes_dir, normals_dir, depth_dir, size, m):
        super(AutoEncoderData, self).__init__()

        self.scenes_dir = scenes_dir
        self.normals_dir = normals_dir
        self.depth_dir = depth_dir
        self.images = sorted(os.listdir(self.scenes_dir))
        self.normals = os.listdir(self.normals_dir)
        self.depths = os.listdir(self.depth_dir)
        self.width = size[0]
        self.height = size[1]
        self.dir = dir
        self.m = m

    def __getitem__(self, index):
        input = torch.zeros(7,self.height, self.width,7)
        output = torch.zeros(7,self.height, self.width,3)
        image = self.images[index]
        splits = image.split('_')
        start = int(splits[2][:-7])
        if start > int(self.m[int(splits[1])] - 7):
            start = int(self.m[int(splits[1])] - 7)
        scene_number = splits[1]
        filenames = []
        for i in range(7):
            filenames.append('scene_'+scene_number+'_'+str(start+i)+'rgb.png')
        for i in range(7):
            image = cv2.imread(self.dir+'/'+'scenes'+'/'+filenames[i])
            image = cv2.resize(image, (256,256))
            gt_filename = 'scene_'+scene_number+'_'+str(int(self.m[int(scene_number)]))+'rgb.png'
            gt = cv2.imread(self.dir+'/'+'scenes'+'/'+gt_filename)
            gt = cv2.resize(gt, (256,256))
            normal_filename = 'scene_'+scene_number+'_'+'normals.png'
            normal = cv2.imread(self.dir+'/'+'normals'+'/'+normal_filename)
            normal = cv2.resize(normal, (256,256))
            depth_filename = 'scene_'+scene_number+'_'+'depth.png'
            depth = cv2.imread(self.dir+'/'+'depths'+'/'+depth_filename,0)
            depth = cv2.resize(depth, (256,256))
            depth = np.expand_dims(depth, axis=2)

            gt = gt.astype(np.float) / 255.0
            image = image.astype(np.float) / 255.0
            normal = normal.astype(np.float) / 255.0
            depth = depth.astype(np.float) / 255.0

            input[i,:,:,:3] = torch.from_numpy(image)
            input[i,:,:,3:6] = torch.from_numpy(normal)
            input[i,:,:,6:7] = torch.from_numpy(depth)

            output[i,:,:,:] = torch.from_numpy(gt)
        return {'image':input.permute(0,3,1,2),'output':output.permute(0,3,1,2)}

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    m = find_max('../test_data/scenes',2)
    #print(np.cumsum(m))
    dataset = AutoEncoderData('../test_data','../test_data/scenes','../test_data/normals','../test_data/depths',(256,256), m)
    data_num = 1
    data = dataset[data_num]
    print(data['image'].shape)
    print(data['output'].shape)
