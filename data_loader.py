# Setting up the dataset for the dataloader

import torch
from torch import nn,optim
from torchvision import transforms,models
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import pandas as pd
import os
from PIL import Image,ImageOps

class DiabeticRetinopathyDataset(Dataset):
    
    """
        Parameters:
        
            csv_file : path to the csv file.
            root_dir : directory with all the images.
            transform : Optional transform to be applied
                on a sample.
    """
    
    def __init__(self,csv_file,root_dir,transform=None):
        self.labels = pd.read_csv(csv_file,encoding = 'UTF-8',engine='python')
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        
        image_name = os.path.join(self.root_dir,self.labels.iloc[idx,0] + '.jpeg')
        image = Image.open(image_name)
        label = self.labels.iloc[idx,1]
        
       
        if self.transform:
            image = self.transform(image)
        
        
        return image,label