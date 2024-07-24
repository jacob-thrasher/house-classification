from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
from torch import nn
import timm
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
import cv2
import os
from torchvision import models
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
import torchvision.transforms as T
from data import ErieParcels
from train_test import train, test_step
import pandas as pd
from utils import create_confusion_matix
import transformers
import random
from torch.optim import Adam, SGD, AdamW

seed = 99
torch.manual_seed(seed)   
np.random.seed(seed)
random.seed(seed)


root = '/home/jacob/Documents/data/erie_data'
model_path = 'vit_base_patch16_224'

for i in range(4, 5):

    train_dataset = ErieParcels(os.path.join(root, 'parcels_cleaned'), os.path.join(root, f'cv/{i}/train.csv'), year_regression=False)
    val_dataset = ErieParcels(os.path.join(root, 'parcels_cleaned'), os.path.join(root, f'cv/{i}/test.csv'), year_regression=False)


    model = timm.create_model(model_path, pretrained=True)
    model.head = nn.Linear(768, 2)

    print(f"STARTING FOLD {i}")
    optim = Adam(model.parameters(), lr=3e-5)
    train(train_dataset, val_dataset, model, optim, model_name='swin_adamW3e-5', epochs=25, show_progress=True)

# print("STARTING MODEL 2")
# optim = Adam(model.parameters(), lr=3e-5, weight_decay=0)
# train(train_dataset, val_dataset, model, optim, model_name='swin_adam3e-5')

# print("STARTING MODEL 3")
# optim = Adam(model.parameters(), lr=1e-4, weight_decay=0)
# train(train_dataset, val_dataset, model, optim, model_name='swin_adam1e-4')

# print("STARTING MODEL 4")
# optim = SGD(model.parameters(), lr=1e-4, momentum=0.9)
# train(train_dataset, val_dataset, model, optim, model_name='swin_sgd1e-4')

# print("STARTING MODEL 5")
# optim = SGD(model.parameters(), lr=1e-3, momentum=0.9)
# train(train_dataset, val_dataset, model, optim, model_name='swin_sgd1e-3')

# print("STARTING MODEL 6")
# optim = SGD(model.parameters(), lr=1e-2, momentum=0.9)
# train(train_dataset, val_dataset, model, optim, model_name='swin_sgd1e-2')

# print("STARTING MODEL 7")
# optim = Adam(model.parameters(), lr=1e-3, weight_decay=0)
# train(train_dataset, val_dataset, model, optim, model_name='swin_adam1e-3')