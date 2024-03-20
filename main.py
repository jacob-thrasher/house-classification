import os
import numpy as np
import torch
import random
import transformers
from torch.optim import Adam, SGD
from torch import nn
from data import ErieParcels
from train_test import train

torch.manual_seed(69)   
np.random.seed(69)
random.seed(69)


root = 'D:\Big_Data'

model = transformers.Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
model.classifier = nn.Linear(768, 2)

optim = Adam(model.parameters(), lr=3e-5, weight_decay=0.01)

train_dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erietrain.csv'), year_regression=False)
val_dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erieval.csv'), year_regression=False)


train(train_dataset, val_dataset, model, optim, model_name='hs_swin_adam3e-5_wd0.01')