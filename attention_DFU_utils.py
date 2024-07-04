# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:02:40 2024

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
import cv2
from vit_keras import visualize
#import random
import os
from skimage import measure, segmentation, draw, morphology
from PIL import Image
import matplotlib.patches as patches 
import pandas as pd
from copy import copy

IMAGE_SIZE = (224,224)


def plot_training(data_hist, val_data_hist):
    

    acc = data_hist[0]
    val_acc = val_data_hist[0]
    loss = data_hist[1]
    val_loss = val_data_hist[1]  
    
    max_acc_train = np.max(acc, axis=0)
    min_acc_train = np.min(acc, axis=0)
    av_acc_train = np.mean(acc, axis=0)
    
    max_acc_val = np.max(val_acc, axis=0)
    min_acc_val = np.min(val_acc, axis=0)
    av_acc_val = np.mean(val_acc, axis=0)
    
    max_loss_train = np.max(loss, axis=0)
    min_loss_train = np.min(loss, axis=0)
    av_loss_train = np.mean(loss, axis=0)
    
    max_loss_val = np.max(val_loss, axis=0)
    min_loss_val = np.min(val_loss, axis=0)
    av_loss_val = np.mean(val_loss, axis=0)

    
    epochs = range(1, len(av_acc_train) + 1)
    
    
    fig2, ax2 = plt.subplots()
    plt.title('Training and Validation Loss')
    plt.plot(epochs, av_loss_train, 'blue', linestyle='--', label='Training loss')
    ax2.fill_between(epochs, min_loss_train, max_loss_train, color = 'blue', alpha=0.2)
    plt.plot(epochs, av_loss_val, 'red', linestyle='-', label='Validation loss')
    ax2.fill_between(epochs, min_loss_val, max_loss_val, color = 'red', alpha=0.2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.grid('True')
    plt.legend()
    
    
    fig3, ax3 = plt.subplots()
    plt.title('Training and Validation Binary Accuracy')
    plt.plot(epochs, av_acc_train, 'red',linestyle='--', label='Training accuracy')
    ax3.fill_between(epochs, min_acc_train, max_acc_train, color = 'red', alpha=0.2)
    
    plt.plot(epochs, av_acc_val, 'blue', linestyle='-', label='Validation accuracy')
    ax3.fill_between(epochs, min_acc_val, max_acc_val, color = 'blue',alpha=0.2)
    
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Metric Score')
    ax3.grid('True')
    plt.legend()
    
    plt.show()
    



def extract_bboxes(imageFolder, maskFolder, bboxFolder_img, bboxFolder_mask, rectFolder, model):
    
    image_files = os.listdir(imageFolder)
    
    names_list = []
    bbox_list = []

        
    for image_file in image_files[:150]:

        image = np.asarray(load_img(imageFolder + image_file))
        image = cv2.resize(image,[224,224])
        image_e = copy(image)
        
        heatmap = visualize.attention_map(model = model.get_layer("vit-b16"), image = image)

        mask = np.asarray(load_img(maskFolder + image_file[:-4] + ".png",target_size=IMAGE_SIZE))
        mask = mask[:,:,0]
    
        edge = cv2.Canny(mask,128,128)
        edge = edge > 0.5
        image_e[edge == 1] = 255
        

        heat_bin = heatmap > 0.75
        heat_bin = heat_bin*1
        heat_bin = np.squeeze(heat_bin)
        heat_labeled = measure.label(heat_bin)
        heat_clear = segmentation.clear_border(heat_labeled)
        heat_big_only = morphology.remove_small_objects(heat_clear, min_size=64)
        heat_bigger = morphology.binary_dilation(heat_big_only)*255

        heat_bigger = measure.label(heat_bigger)
        props = measure.regionprops(heat_bigger)
                
        fig, ax = plt.subplots(1) 
        ax.imshow(image_e)
        ax.imshow(heatmap, cmap='jet', alpha=0.3)
                
        for i in range(len(props)):
            
            bbox = props[i].bbox
            
            names_list.append(image_file)
            bbox_list.append(bbox)            
            
            print(bbox)
            img_box = image[bbox[0]:bbox[2],bbox[1]:bbox[3]]
            mask_box = mask[bbox[0]:bbox[2],bbox[1]:bbox[3]]
            Image.fromarray(img_box).save(bboxFolder_img + image_file[:-4] + "_" + str(i) + ".png")
            Image.fromarray(mask_box).save(bboxFolder_mask + image_file[:-4] + "_" + str(i) + ".png")
            rect = patches.Rectangle((bbox[1],bbox[0]),bbox[3]-bbox[1], bbox[2]-bbox[0], linewidth=1, edgecolor='r', facecolor="none") 
            ax.add_patch(rect) 
        fig.show() 
        fig.savefig(rectFolder + image_file, bbox_inches='tight')
        
    dict = {'file': names_list, 'bbox': bbox_list}
    df = pd.DataFrame(dict)
        
    df.to_csv('bbox_coords.csv', index=False) 
        

    
    
    
    