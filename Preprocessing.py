# Preprocessing Pipeline

import numpy as np
import pandas as pd
import os
from PIL import Image,ImageOps
import cv2
from os.path import abspath, realpath
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import shutil



# https://kaggle2.blob.core.windows.net/forum-message-attachments/88655/2795/competitionreport.pdf

def ScaleRadius(image,scale):
    
    """ Scales Radius of image
        
        Parameters: 
            image (matplotlib image) : a matplotlib image
            Scale (integer) : the scaling fcator
               
        Returns: 
            (scaled_image) : the scaled image
    """
    
    x = image[image.shape[0] // 2,:,:].sum(1)
    rx = (x > x.mean() / 10).sum() / 2
    
    y = image[:,image.shape[1] // 2,:].sum(1)
    ry = (y > y.mean() / 10).sum() / 2
    
    scale_factor = scale * 1.0 / max(rx,ry)
    
    scaled_image = cv2.resize(image,(0,0),fx = scale_factor,fy=scale_factor)
    
    return scaled_image


def crop_and_resize(image,image_size=512):
        
        """ Crops and resizes the image
        
        Parameters: 
            image (matplotlib image) : a matplotlib image
            image_size (integer) : the final size to which image is resized
               
        Returns: 
            (new_image) : the cropped and resized image
        """
    
        image = Image.fromarray(image)
        width,height = image.size
    
        if width > height:
            new_width = height
            lrcrop = int((width-new_width) / 4)
            crop_extra_pixel = (width-new_width) % 2
            cropped_image = image.crop((lrcrop,0,width - lrcrop - crop_extra_pixel,height))
    
        elif height > width:
            new_height = width
            tbcrop = int((height-new_height) / 4)
            crop_extra_pix = (height-new_height) % 2
            cropped_image = image.crop((0,tbcrop,width,height - tbcrop - crop_extra_pix))
    
        else:
            cropped_image = image
    
        new_image = cropped_image.resize((image_size,image_size),Image.ANTIALIAS)
    
        return new_image
    

def SubtractColor(image,scale):
    
         """  remove noise in the images and perform image blending
        
        Parameters: 
            image (matplotlib image) : a matplotlib image
            Scale (integer) : the scaling fcator
               
        Returns: 
            (scaled_image) : the modified image
            
         """
    
         image = np.asarray(image)
        
         gb_image = cv2.GaussianBlur(image,(0,0),int(scale/30))
    
         modified_image = cv2.addWeighted(image,4,gb_image,-4,128)
    
         return modified_image
    

def Clahe(image):
    
    """ Applied contast limiting adaptive histogram equalization(CLAHE)
     
        Parameters: 
            image (PIL image) : a PIL image
            
        Returns: 
            (new_image) : image after clahe
    """
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_lab_planes = cv2.split(image)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image_lab_planes[0] = clahe.apply(image_lab_planes[0])
    
    modified_image = cv2.merge(image_lab_planes)
    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_LAB2RGB)
    
    return modified_image
    
    
def create_directory(directory):
    """
        Parameters: 
            directory  : name of directory to create
            
        Returns: 
            creates a new directory
    """
    if directory.is_dir():
        shutil.rmtree(directory)
        
    os.makedirs(directory)
    
    
def preprocessing_pipeline():
    
    """
        returns a new directory with processed images
    """
    
    new_directory_path = Path('C:\\Users\\prani\\Desktop\\train\\Processed_imagestmp\\')
    create_directory(new_directory_path)
    file_path = Path('C:\\Users\\prani\\Desktop\\train\\Temp\\')
    
    if file_path.is_dir:
        files = os.listdir(file_path)

        for file in files:
            image = mpimg.imread(os.path.join(file_path,file))
            processed_image = ScaleRadius(image,scale = 300)
            processed_image = crop_and_resize(processed_image,image_size=299)
            processed_image = SubtractColor(processed_image,scale=300)
            processed_image = Clahe(processed_image)
            modified_image = Image.fromarray(processed_image)
            modified_image.save(os.path.join(new_directory_path,file))
    else:
        raise ValueError('The main Directory doesnt exist')
        