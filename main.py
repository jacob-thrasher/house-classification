import os
import numpy as np
import torch
import random
import transformers
import pytorch_warmup as warmup
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from torch.optim import Adam, SGD
from torch import nn
from data import ErieParcels
from train_test import train

torch.manual_seed(69)   
np.random.seed(69)
random.seed(69)


config = {
    'model_name': '',
    'epochs': 10,
    'batch_size': 32,
    'accum_iter': 1,
    'warmup_epochs': 1,
    'lr': 3e-5,
    'weight_decay': 0.01
}

root = 'D:\Big_Data'

model = transformers.Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
model.classifier = nn.Linear(768, 2)

optim = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

train_dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erietrain.csv'), year_regression=False)
val_dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erieval.csv'), year_regression=False)

train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False) 


iter_per_epoch = int(len(train_dataloader) / config['accum_iter'])
warmup_period = config['warmup_epochs'] * iter_per_epoch
warmup_scheduler = warmup.LinearWarmup(optim, warmup_period=warmup_period)
cosine_scheduler = CosineAnnealingLR(optim, T_max=iter_per_epoch*(config['epochs'] - config['warmup_epochs']))

train(train_dataloader, val_dataloader, model, optim, config)