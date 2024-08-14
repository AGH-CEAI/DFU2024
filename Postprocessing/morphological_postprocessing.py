# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 17:27:31 2024

@author: user
"""

inFolder = "C:/Users/user/Desktop/DFU_attentions/sam_Bartek_Darek/in/"
outFolder = "C:/Users/user/Desktop/DFU_attentions/sam_Bartek_Darek/out/"
        
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import os
from skimage import morphology
from PIL import Image

image_files = os.listdir(inFolder)

i = 0
       
for image_file in image_files:
    
    i = i + 1 
    print(i)

    mask = np.asarray(load_img(inFolder + image_file))
    mask = mask[:,:,0]

    SE = morphology.disk(2)

    mask = mask > 0.5
    mask = morphology.binary_closing(mask, footprint = SE)
    mask = morphology.binary_opening(mask, footprint = SE)
    mask= morphology.remove_small_objects(mask, min_size=64)
    
    mask = (mask*255).astype('uint8')

    Image.fromarray(mask).save(outFolder + image_file[:-4] + ".png")
    
                