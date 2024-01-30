import json
from time import sleep
import pandas as pd
from data import ErieParcels
import matplotlib.pyplot as plt
import os
from network import train
import torch
from utils import create_confusion_matix
from torch.utils.data import DataLoader
from network import SimpleCNN
root = 'D:\\Big_Data'
# csvpath = 'D:\\Big_Data\\parcel_info.csv'
# df = pd.read_csv(csvpath)
# train = df.sample(frac=0.8)
# test = df[~df.index.isin(train.index)]

# print(len(train), len(test))

# train.to_csv(os.path.join(dst, 'erietrain.csv'))
# test.to_csv(os.path.join(dst, 'erietest.csv'))

# dataset = ErieParcels(dataroot='D:\\Big_Data\\parcels', csvpath=csvpath)


train_dataset = ErieParcels(dataroot=os.path.join(root, 'parcels'), csvpath=os.path.join(root, 'erietrain.csv'))
test_dataset = ErieParcels(dataroot=os.path.join(root, 'parcels'), csvpath=os.path.join(root, 'erietest.csv'))

# test_dataloader = DataLoader(test_dataset, batch_size=128)

# model = SimpleCNN()
# model.load_state_dict(torch.load('models\\homestead-status.pt'))
# device = 'cuda'
# model.to(device)
# create_confusion_matix(model, test_dataloader, device)


train(train_dataset, test_dataset, model_name='homestead-status')


