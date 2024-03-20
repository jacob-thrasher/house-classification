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

torch.manual_seed(69)   
np.random.seed(69)
random.seed(69)


root = '/home/jacob/Documents/data/erie_data'
model_path = "microsoft/swinv2-tiny-patch4-window8-256"

train_dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erietrain.csv'), year_regression=False, model_path=model_path)
val_dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erieval.csv'), year_regression=False, model_path=model_path)


model = transformers.Swinv2ForImageClassification.from_pretrained(model_path)
model.classifier = nn.Linear(768, 2)

print("STARTING MODEL 1")
optim = AdamW(model.parameters(), lr=3e-5)
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