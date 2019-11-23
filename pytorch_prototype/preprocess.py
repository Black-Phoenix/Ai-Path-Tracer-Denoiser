import numpy as np
import torch, os, sys, cv2

def preprocess(root_dir, scenes_dir,normals_dir, depth_dir, albedos_dir, m):
    images = sorted(os.listdir(scenes_dir))
    inputs = np.zeros((len(images),256,256,10))
    outputs = np.zeros((len(images),256,256,3))
    for index in range(len(images)):
        print(index)
        image = cv2.imread(scenes_dir+'/'+images[index])
        image = cv2.resize(image, (256,256))

        splits = images[index].split('_')

        gt_filename = splits[0]+'_'+'0008.png'
        gt = cv2.imread(scenes_dir+'/'+gt_filename)
        gt = cv2.resize(gt, (256,256))

        normal = cv2.imread(normals_dir+'/'+splits[0]+'.png')
        normal = cv2.resize(normal, (256,256))

        albedo = cv2.imread(albedos_dir+'/'+splits[0]+'.png')
        albedo = cv2.resize(albedo, (256,256))

        depth = cv2.imread(depth_dir+'/'+splits[0]+'.png',0)
        depth = cv2.resize(depth, (256,256))
        depth = np.expand_dims(depth, axis=2)

        gt = gt.astype(np.float) / 255.0
        image = image.astype(np.float) / 255.0
        normal = normal.astype(np.float) / 255.0
        depth = depth.astype(np.float) / 255.0

        inputs[index,:,:,:3] = image
        inputs[index,:,:,3:6] = normal
        inputs[index,:,:,6:7] = depth
        inputs[index,:,:,7:10] = albedo

        outputs[index] = gt

    return inputs,outputs
