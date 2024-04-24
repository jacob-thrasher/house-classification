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

def get_train_test(root, csvpath, test_split=.1):
    df = pd.read_csv(csvpath)
    files = list(df['file'])
    random.shuffle(files)
    files.remove('file')
    test_size = int(len(files) * test_split)
    
    train_files = files[test_size:]
    test_files = files[:test_size]

    train_dataset = ZillowSupervised(root, train_files, csvpath)
    test_dataset = ZillowSupervised(root, test_files, csvpath)
    return train_dataset, test_dataset

class ErieParcels(Dataset):
    def __init__(self, dataroot, csvpath, img_dim=224, year_regression=False, img_processor=None):
        self.dataroot = dataroot
        self.df = pd.read_csv(csvpath)

        self.df = self.df.loc[self.df['parcel_number'].isin(os.listdir(dataroot))]

        if year_regression: self.df = self.df[~self.df['year_built'].isna()]
        else:
            self.df = self.df[self.df['classification'] != 'E']
            self.df = self.df[self.df['classification'] != 'F']

        self.augment = T.Compose([
            T.RandomCrop(img_dim),
            T.RandomHorizontalFlip(),
            T.ColorJitter()
        ])

        if img_processor is not None: self.processor = img_processor
        else: self.processor = T.Compose([
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        parcel_number = str(row['parcel_number'])
        img = Image.open(os.path.join(self.dataroot, parcel_number, '0.png')).convert("RGB")
        # img = self.preprocess(img)
        # img = self.processor(img, return_tensors='pt')
        # img = self.processor(img)

        homestread_status = 1 if row['homestead_status'] == 'Inactive' else 0
        year = row['year_built']
        # return img, math.floor(abs(float(year)))
        # return img, int(abs(float(year)) > 1971)
        # return self.augment(img.pixel_values.squeeze()), homestread_status
        return T.functional.resize(img, (224, 224)), homestread_status, parcel_number

class ZillowSupervised(Dataset):
    def __init__(self, root, files, csvpath, img_dim=224):
        preprocess = T.Compose([
            T.Resize(img_dim),file
        ])

        self.df = pd.read_csv(csvpath)
        self.imgs = []
        self.files = files
        for file in tqdm(self.files):
            img = Image.open(os.path.join(root, file)).convert("RGB")
            img = preprocess(img)
            self.imgs.append(img)
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]

        key = self.files[idx]
        label = self.df[self.df['file'] == key].reset_index()['label'][0] # There is def a better way to do this
        return img, int(label)

class ZillowUnsupervised(Dataset):
    def __init__(self, root, img_dim=224):

        self.preprocess = T.Compose([
            T.Resize(img_dim),
            T.CenterCrop(img_dim),
            T.ToTensor(),
            T.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        self.root = root
        self.files = os.listdir(root)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.files[idx])).convert('RGB')
        return self.preprocess(img), self.files[idx]
    
def cluster(root, n=2):
    dataset = ZillowUnsupervised(root)

    # Temp soln
    imgs = []
    for i in range(0, 500):
        img, name = dataset[i]
        imgs.append(img)
    imgs = torch.stack(imgs).to('cuda')

    inception = timm.create_model('inception_v4', pretrained=True).to('cuda')
    embeddings = inception(imgs)

    kmeans = KMeans(n_clusters=2, init='random')    
    print("Fitting data")
    kmeans.fit(embeddings.cpu())

    print("Making predictions")
    Z = kmeans.predict(imgs)

    plot_clusters(Z, imgs)

def plot_clusters(Z, data):
    for i in range(0,2):
        row = np.where(Z==i)[0]  # row in Z for elements of cluster i
        num = row.shape[0]       #  number of elements for each cluster
        r = np.floor(num/10.)    # number of rows in the figure of the cluster 

        print("cluster "+str(i))
        print(str(num)+" elements")

        plt.figure(figsize=(10,10))
        for k in range(0, num):
            plt.subplot(r+1, 10, k+1)
            image = data[row[k], ]
            image = image.reshape(8, 8)
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.show()

# cluster('D:\\Big_Data\\zillow_images')

# inception = timm.create_model('inception_v4', pretrained=True)
# inception.last_linear = nn.Identity()
# print(inception)

# root = 'D:\\Big_Data\\zillow_images'
# csvpath = 'labels.csv'

# train_dataset, test_datset = get_train_test(root, csvpath)

# # print(len(train_dataset), len(test_datset))
# img, label = train_dataset[0]

# print(label)
# plt.imshow(img.permute(1, 2, 0))
# plt.show()