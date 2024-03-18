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
from network import train, test_step
import pandas as pd
from utils import create_confusion_matix
import transformers
import random
from torch.optim import Adam, SGD

torch.manual_seed(69)   
np.random.seed(69)
random.seed(69)


root = 'D:\Big_Data'

train_dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erietrain.csv'), year_regression=False)
val_dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erieval.csv'), year_regression=False)


model = transformers.Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
model.classifier = nn.Linear(768, 2)

print("STARTING MODEL 1")
optim = Adam(model.parameters(), lr=3e-5, weight_decay=0.01)
train(train_dataset, val_dataset, model, optim, model_name='hs_swin_adam3e-5_wd0.01')

print("STARTING MODEL 2")
optim = Adam(model.parameters(), lr=3e-5, weight_decay=0)
train(train_dataset, val_dataset, model, optim, model_name='hs_swin_adam3e-5')

print("STARTING MODEL 3")
optim = Adam(model.parameters(), lr=1e-4, weight_decay=0)
train(train_dataset, val_dataset, model, optim, model_name='hs_swin_adam1e-4')

print("STARTING MODEL 4")
optim = SGD(model.parameters(), lr=1e-4, momentum=0.9)
train(train_dataset, val_dataset, model, optim, model_name='hs_swin_sgd1e-4')