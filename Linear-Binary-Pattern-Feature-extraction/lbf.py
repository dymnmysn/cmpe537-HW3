# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 21:47:29 2021

@author: krgza

###############################################################################################
        Input: 
        --Falder path
            Give path to "folder" variable as shown below 
            
            ======> folder = "#####PATH TO IMAGES FOLDER####### /Caltech20"  <========
            
            And give subfolder name training/testing.
            Then run code.
        
        Outputs:
            -- {name of image class}_subfoldername.csv
            
                hist_airplanes_training.csv
                            OR
                hist_airplanes_training.csv
                
                Each column of output file represents one image's features.
                
###############################################################################################
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os 
from tqdm import tqdm

def pixel_calculator(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):   
    center = img[x][y]
    val_ar = []
    val_ar.append(pixel_calculator(img, center, x-1, y+1))     # top_right
    val_ar.append(pixel_calculator(img, center, x, y+1))       # right
    val_ar.append(pixel_calculator(img, center, x+1, y+1))     # bottom_right
    val_ar.append(pixel_calculator(img, center, x+1, y))       # bottom
    val_ar.append(pixel_calculator(img, center, x+1, y-1))     # bottom_left
    val_ar.append(pixel_calculator(img, center, x, y-1))       # left
    val_ar.append(pixel_calculator(img, center, x-1, y-1))     # top_left
    val_ar.append(pixel_calculator(img, center, x-1, y))       # top
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val    

def hist_calculator(file_name):
    img_bgr = cv2.imread(file_name)
    height, width, channel = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    img_lbp = np.zeros((height, width,3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
             img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
    hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])

    return hist_lbp

if __name__ == '__main__':
    
    folder = "#####PATH TO IMAGES FOLDER#######\\Caltech20"
    subfolder = "training" #testing
    folder_path = os.path.join(folder,subfolder)
    for category in (os.listdir(folder_path)): 
        path = os.path.join(folder_path,category)
        histograms = np.zeros((256,1,len(os.listdir(path))))
        #histograms = []
        histname = "hist_{}_{}.csv".format(category,subfolder)
        running_bar=tqdm(enumerate(os.listdir(path)), total=len(os.listdir(path)))
        for num,img_path in running_bar:
            #print(img_path)
            img_path = os.path.join(path,img_path)
            #print(img_path)
            histograms[:,:,num]=np.array(hist_calculator(img_path))
            #if num == 10:
                #break
            
        #running_bar.set_description("Running {}".format(histname))
        histograms = np.squeeze(histograms)
        df = pd.DataFrame(histograms) 
        df.to_csv(histname)
        #break
        
