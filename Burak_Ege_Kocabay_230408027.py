# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 13:33:05 2025

@author: burakegekocabay
"""
import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops


def find_paths(path): #finding all images in datasets
    datasets = []
    extensions = (".jpg",".jpeg",".png",".webp") #valid extensions for images
    
    for dir_name in os.listdir(path):
        dir_path = os.path.join(path, dir_name)
        for file_name in os.listdir(dir_path):
            if(file_name.endswith(extensions)):
                file_path = os.path.join(dir_path, file_name)
                datasets.append((file_path,dir_name))
    return datasets #save all images path

# --------------------------------------------------------------------------------------------------

def data_and_features(datasets): 
    data = []
    for image_path,class_name in datasets:
        gray = convert_to_gray(image_path) #converting to gray beauce glcm only works in gray
        if gray is None:
            continue
        features = process_glcm(gray)  #sending gray images for glcm one by one 
        features["class"] = class_name #adding classname(cat,dog,car... to features)
        data.append(features)
        
    return create_dataframe(data)

def convert_to_gray(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def create_dataframe(data):
    return pd.DataFrame(data)
        

def process_glcm(gray):
    feature_names = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    features = {}
    
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        symmetric=True, 
        normed=True
    )
    
    for name in feature_names:
        values = graycoprops(glcm, name)
        features[name] = np.mean(values)  # average over 4 directions
        
    features["std"] = np.std(gray) #standard deviation for extra feauture
    return features
    
# -------------------------------------------------------------------------------------------------------
def dataframe_to_arrf(dataframe):
    filename="glcm_features.arff"
    
    with open(filename, "w") as f:
        f.write("@RELATION glcm_features\n\n")
        for col in dataframe.columns[:-1]:
            f.write(f"@ATTRIBUTE {col} NUMERIC\n") #write numerical values from dataframe
        
        classes = ",".join(sorted(dataframe["class"].unique()))
        f.write(f"@ATTRIBUTE class {{{classes}}}\n\n")
        f.write("@DATA\n")
        for _, row in dataframe.iterrows():
            f.write(",".join(map(str, row.values)) + "\n")
    print(f"ARFF file saved: {filename}")
    
    
    
if __name__ == "__main__":
    path = "img"  # DataSet Directory : /img/......
    
    datasets = find_paths(path) #all image paths
    dataframe = data_and_features(datasets)
    dataframe_to_arrf(dataframe)