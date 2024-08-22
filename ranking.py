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
val_masks = os.path.join(root, 'active_learning/masks')
pretrained_model_path = 'figures/AL-entropy/iter_45/best_model.pt'
model_path = 'vit_base_patch16_224'
model_checkpoint = '/users/jdt0025/timm_models/vit.pt'

# categories = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 
#             'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
static_categories = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky']


val_dataset = ErieParcels_top4s(os.path.join(root, 'parcels_cleaned'), os.path.join(root, 'active_learning/active_learning/top-4s.csv'), split=split)
dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=True)
total_images = len(val_dataset)  

model = timm.create_model(model_path, checkpoint_path=model_checkpoint)

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

metrics = {
    'mAP': 0,
    'NDCG': 0,
    'RPREC': 0
}
global_scores = {}
for c in static_categories: global_scores[c] = 0

for imgs, info_batch in tqdm(dataloader, disable=False):
    imgs = imgs.to('cuda')

    # Get attention and load masks
    grayscale_cam = cam(input_tensor=imgs, targets=target)
    masks = load_batch_mask(val_masks, info_batch['parcel_number'])

    dict_list = []
    for i in range(len(info_batch)):
        this_dict = {
            'homestead_status': info_batch['homestead_status'][i],
            'parcel_number': info_batch['parcel_number'][i],
            'top4': [x[i] for x in info_batch['top4']]
        }
        dict_list.append(this_dict)


    imgs = imgs.cpu()
    for mask, attn, img, info in zip(masks, grayscale_cam, imgs, dict_list):
        # Check if house in mask
        unique, counts = np.unique(mask, return_counts=True)
        cat_counts = dict(zip(unique, counts))
        if 3 not in list(cat_counts.keys()): # Building
            total_images -= 1
            continue # Skip if no buildling

        # Score for this image
        image_scores = {}
        for c in static_categories: global_scores[c] = 0

        # print(np.min(mask))
        # plt.imshow(mask)
        # plt.savefig('test_imgs/mask.png')
        # plt.close()

        for i in range(len(static_categories)):
            category = static_categories[i]
            category_mask = mask.copy()

            category_mask = np.where(category_mask == (i+1), category_mask, 0)
            category_mask /= (i+1) # Set values to 1


            # plt.imshow(category_mask, cmap='jet')
            # plt.savefig(f'test_imgs/{category}_mask.png')
            # plt.close()

            weighted_sum = np.sum(attn * category_mask)
            total_nonzero = np.count_nonzero(category_mask)

            if total_nonzero > 0:
                normalized_sum = weighted_sum / total_nonzero
            else: normalized_sum = 0

            global_scores[category] += normalized_sum
            image_scores[category] = normalized_sum

        relevent_items = list(info['top4'])
        mAP, ndcg, rprec = get_retrieval_metrics(image_scores, relevent_items, k=k, threshold=threshold)
        metrics['mAP'] += mAP
        metrics['NDCG'] += ndcg
        metrics['RPREC'] += rprec

        sorted_scores = get_sorted_scores(image_scores)
        title = 'Pred: ' + str(sorted_scores[0][:4]) + '\nGT: ' + str(relevent_items)
        status_label = 'active' if info['homestead_status'] == 0 else 'inactive'
        # plt.imshow(img.permute(1, 2, 0))
        # plt.imshow(attn*category_mask, cmap='jet', alpha=0.7)
        # plt.title(title)
        # plt.xlabel(f'Status: {status_label} | mAP: {mAP}')
        # plt.savefig(f"test_imgs/{info['parcel_number']}.png")
        # plt.close()

# Average scores
for key in global_scores: global_scores[key] /= total_images
total_skipped = len(val_dataset) - total_images
print("Total images skipped:", total_skipped)

scores = dict(sorted(global_scores.items(), key=lambda item: item[1]))
pprint.pp(scores)

print(' mAP:', metrics['mAP'] / total_images)
print('RPrec:', metrics['RPREC'] / total_images)
print('NDCG:', metrics['NDCG'] / total_images)