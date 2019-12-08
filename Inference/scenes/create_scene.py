# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:10:41 2019

@author: dewan
"""
import random
import numpy as np

for i in range(1,30):
    f_read = open("scene_1.txt","r")
    f_write = open("scene_{}.txt".format(i+1),"w")
    material = 0
    obj = 0
    for x in f_read:
        flag = 0
        if material >= 5:                           
            if "REFL" in x:
                refl = random.uniform(0, 1)
                x = "REFL\t{}\n".format(refl)
            elif "REFRIOR" in x:
                refrior = random.uniform(0,2)
                x = "REFRIOR\t{}\n".format(refrior)
            elif "REFR" in x:
                refr = 0.97 - refl
                x = "REFR\t{}\n".format(refr)
            elif "EMITTANCE" in x:
                emittance = np.random.choice(np.arange(0,5),p=[0.8,0.05,0.05,0.05,0.05])
                x = "EMITTANCE\t{}\n".format(emittance)
            elif "SPECRGB" in x:
                r = random.uniform(0,1)
                g = random.uniform(0,1)
                b = random.uniform(0,1)
                x = "SPECRGB\t{} {} {}\n".format(r, g, b)                
            elif "RGB" in x:
                r = random.uniform(0,1)
                g = random.uniform(0,1)
                b = random.uniform(0,1)
                x = "RGB\t{} {} {}\n".format(r, g, b)
        if obj >= 7:
            if "TRANS" in x:
                trans_x = random.uniform(-4,4)
                trans_y = random.uniform(0,4)
                trans_z = random.uniform(-4,4)
                x = "TRANS\t{} {} {}\n".format(trans_x, trans_y, trans_z)
            elif "SCALE" in x:
                scale_x = random.uniform(1,4)
                scale_y = random.uniform(1,4)
                scale_z = random.uniform(1,4)
                x = "SCALE\t{} {} {}\n".format(scale_x, scale_y, scale_z)
            elif "ROTAT" in x:
                rotat_x = random.uniform(-45,45)
                rotat_y = random.uniform(-45,45)
                rotat_z = random.uniform(-45,45)
                x = "ROTAT\t{} {} {}\n".format(rotat_x, rotat_y, rotat_z)
        if "EYE" in x:
            eye_x = random.uniform(-2,2)
            eye_y = random.uniform(2,8)
            eye_z = random.uniform(8,12)
            x = "EYE\t{} {} {}\n".format(eye_x, eye_y, eye_z)     
        f_write.write(x)
        if "MATERIAL" in x:
                material += 1
        if "OBJECT" in x:
                obj += 1
    f_write.close()