import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
import numpy as np
import gc
from tqdm import tqdm
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from transformers import SamModel, SamProcessor, pipeline
from data import ErieParcels


root = 'D:\\Big_Data\\Erie'


dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erieval.csv'), return_id=True)

generator = pipeline('image-segmentation', model="nvidia/segformer-b1-finetuned-cityscapes-1024-1024", device=0)


categories = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 
            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


counts = {
    'active': 0,
    'total_active': 0,
    'inactive': 0,
    'total_inactive': 0
}
for X, label, parcel_number in tqdm(dataset):
    outputs = generator(X, points_per_batch=64)

    predicted_classes = [x['label'] for x in outputs]

    status = 'active' if label == 0 else 'inactive'

    counts[f'total_{status}'] += 1

    if 'building' in predicted_classes: 
        counts[status] += 1


print(f"ACTIVE: {counts['active']} / {counts['total_active']} =", counts['active'] / counts['total_active'])
print(f"INACTIVE: {counts['inactive']} / {counts['total_inactive']} =", counts['inactive'] / counts['total_inactive'])

