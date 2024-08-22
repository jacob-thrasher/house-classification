import os
import numpy as np
import torch
import timm
import matplotlib.pyplot as plt
import pprint
from torch import nn
from collections import OrderedDict
from tqdm import tqdm
from PIL import Image
from torchmetrics.classification import BinaryJaccardIndex
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM, ScoreCAM, EigenCAM, XGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from data import ErieParcels, ErieParcels_top4s
from utils import *

# models: 
# VGG16            : 'vgg16.tv_in1k'
# ResNet50         : 'resnet50.a1_in1k'
# DenseNet121      : 'densenet121.tv_in1k'
# ViT              : 'vit_base_patch16_224'
# SwinTransformerV1: 'swin_large_patch4_window7_224.ms_in22k'
# SwinTransformerV2: 'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k'

# Testing parameters
k = 4
split = None
threshold = 0

##########

root = '/users/jdt0025/scratch/Erie'
pretrained_model_path = 'figures/AL-ratio/iter_45/best_model.pt'
model_path = 'vit_base_patch16_224'

# categories = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 
#             'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
static_categories = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky']


val_dataset = ErieParcels(os.path.join(root, 'parcels_cleaned'), os.path.join(root, f'active_learning/active_learning/test.csv'), year_regression=False, return_id=True, split='Inactive')
dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=True)
total_images = len(val_dataset)  

model = timm.create_model(model_path, pretrained=True)

if model_path == 'vgg16.tv_in1k':
    model.head.fc = nn.Sequential(nn.Linear(4096, 1000), 
                                  nn.ReLU(),
                                  nn.Linear(1000, 2))
elif model_path == 'resnet50.a1_in1k':
    model.fc = nn.Sequential(nn.Linear(2048, 1000), 
                            nn.ReLU(),
                            nn.Linear(1000, 2))
    target_layers = [model.layer4[-1]]
elif model_path == 'vit_base_patch16_224':
    model.head = nn.Linear(768, 2)
    target_layers = [model.blocks[-1].norm1]
elif model_path == 'swin_large_patch4_window7_224.ms_in22k':
    model.head.fc = nn.Linear(1536, 2)

state_dict = torch.load(pretrained_model_path)
new_dict = OrderedDict()
for key in state_dict:
    value = state_dict[key]
    if 'module' in key:
        key = key.replace('module.', '')

    new_dict[key] = value
model.load_state_dict(new_dict)
model.eval()
model.to('cuda')

cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) # Reshape transform for ViT
target = None

for img, label, parcel_number in tqdm(val_dataset, disable=False):
    img = img.to('cuda')

    pred = model(img.unsqueeze(0))
    pred = torch.argmax(pred, dim=1).item()

    # Get attention and load masks
    grayscale_cam = cam(input_tensor=img.unsqueeze(0), targets=target)

    pred = 'active' if pred == 0 else 'inactive'
    label = 'active' if label == 0 else 'inactive'

    plt.imshow(img.permute(1, 2, 0).cpu())
    plt.imshow(grayscale_cam[0, :], alpha=0.62)
    plt.title(f'{parcel_number}: Pred: {pred}, Label: {label}')
    plt.savefig(f'cam/{parcel_number}.png')
    plt.close()
    # viz = show_cam_on_image(img, grayscale_cam[0, :], use_rgb=True)