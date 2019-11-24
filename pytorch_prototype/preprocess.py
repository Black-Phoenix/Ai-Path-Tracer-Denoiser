import numpy as np
import torch, os, sys, cv2

def preprocess(root_dir, scenes_dir,normals_dir, depth_dir, albedos_dir, gt_dir, m):
    images = sorted(os.listdir(scenes_dir))
    normals = sorted(os.listdir(normals_dir))
    depths = sorted(os.listdir(depth_dir))
    albedos = sorted(os.listdir(albedos_dir))
    gts = sorted(os.listdir(gt_dir))
    inputs = np.zeros((len(images),256,256,10))
    outputs = np.zeros((len(images),256,256,3))
    for index in range(len(images)):
        image = cv2.imread(scenes_dir+'/'+images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256,256))

        gt = cv2.imread(scenes_dir+'/'+gts[index])
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        gt = cv2.resize(gt, (256,256))

        normal = cv2.imread(normals_dir+'/'+normals[index])
        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
        normal = cv2.resize(normal, (256,256))

        albedo = cv2.imread(albedos_dir+'/'+albedos[index])
        albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
        albedo = cv2.resize(albedo, (256,256))

        depth = cv2.imread(depth_dir+'/'+depths[index],0)
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
