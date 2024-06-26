import os
import torch
from torch import nn
import transformers
import torchvision.transforms as T
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import timm
import pandas as pd
import random
import math


class ErieParcels(Dataset):
    def __init__(self, dataroot, csvpath, img_dim=224, year_regression=False, split=None, return_id=False):
        assert split in ['Active', 'Inactive', None], f'Expected split to be in [Active, Inactive], got {split}'

        self.dataroot = dataroot
        self.df = pd.read_csv(csvpath)
        self.return_id = return_id

        valid_parcels = [x.split('.')[0] for x in os.listdir(dataroot)]
        self.df = self.df.loc[self.df['parcel_number'].isin(valid_parcels)]

        if split: self.df = self.df[self.df['homestead_status'] == split]
        if year_regression: self.df = self.df[~self.df['year_built'].isna()]
        else:
            self.df = self.df[self.df['classification'] != 'E']
            self.df = self.df[self.df['classification'] != 'F']

        self.augment = T.Compose([
            T.Resize((img_dim, img_dim)),
            T.ToTensor(),
            T.Normalize((0), (1))
            # T.RandomCrop(img_dim), # Not doing anything
            # T.RandomHorizontalFlip(),
            # T.ColorJitter()
        ])

        # self.swin_processor = transformers.AutoImageProcessor.from_pretrained(model_path)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        parcel_number = str(row['parcel_number'])
        img = Image.open(os.path.join(self.dataroot, f'{parcel_number}.png')).convert("RGB")
        # img = self.preprocess(img)
        # img = self.swin_processor(img, return_tensors='pt')
        
        homestread_status = 1 if row['homestead_status'] == 'Inactive' else 0
        year = row['year_built']
        # return img, math.floor(abs(float(year)))
        # return img, int(abs(float(year)) > 1971)
        # return self.augment(img.pixel_values.squeeze()), homestread_status
        if self.return_id: return self.augment(img), homestread_status, parcel_number
        return self.augment(img), homestread_status
        # return img, homestread_status
    
class ErieParcels_top4s(Dataset):
    def __init__(self, dataroot, csvpath, img_dim=224, split=None):
        self.dataroot = dataroot
        self.df = pd.read_csv(csvpath)

        if split: self.df = self.df[self.df['homestead_status'] == split]

        valid_parcels = [x.split('.')[0] for x in os.listdir(dataroot)]
        self.df = self.df.loc[self.df['parcel_number'].isin(valid_parcels)]
        # self.df = self.df.loc[self.df['parcel_number'].isin(os.listdir(dataroot))]

        self.augment = T.Compose([
            T.Resize((img_dim, img_dim)),
            T.ToTensor(),
            T.Normalize((0), (1))
        ])

    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        parcel_number = str(row['parcel_number'])
        # img = Image.open(os.path.join(self.dataroot, parcel_number, '0.png')).convert("RGB")
        img = Image.open(os.path.join(self.dataroot, f'{parcel_number}.png')).convert("RGB")

        homestead_status = 1 if row['homestead_status'] == 'Inactive' else 0

        top4 = [row['top1'], row['top2'], row['top3'], row['top4']]

        info = {
            'homestead_status': homestead_status,
            'parcel_number': parcel_number,
            'top4': top4
        }

        return self.augment(img), info
        # return T.functional.resize(img, (224, 224)), info