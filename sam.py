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

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    print(mask_image)
    ax.imshow(mask_image)
    del mask
    gc.collect()


def show_masks_on_image(raw_image, masks):
  plt.imshow(np.array(raw_image))
  ax = plt.gca()
  ax.set_autoscale_on(False)
  for mask in masks:
      show_mask(mask, ax=ax, random_color=True)
  plt.axis("off")
  plt.savefig('test.png')
  del mask
  gc.collect()

root = '/home/jacob/Documents/data/erie_data'


dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erieval.csv'), img_processor=None)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

generator = pipeline('image-segmentation', model="nvidia/segformer-b1-finetuned-cityscapes-1024-1024", device=0)


categories = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 
            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


for X, _, parcel_number in tqdm(dataset):
    outputs = generator(X, points_per_batch=64)
    overlap_checking = []
    all_masks = []
    for i, category in enumerate(outputs):
        mask = np.array(category['mask'], dtype=int) / 255
        overlap_checking.append(mask)
        mask = mask * (categories.index(category['label']) + 1)
        all_masks.append(mask)
    if np.max(sum(overlap_checking)) > 1: raise ValueError('Found overlapping masks!!!')

    summed_masks = sum(all_masks)
    if np.max(summed_masks) > 20: raise ValueError("")
    summed_masks_normalized = summed_masks / 20 # number of classes

    pil_img = Image.fromarray(summed_masks_normalized*255).convert('RGB')

    pil_img.save(os.path.join(root, f'masksval_224/{parcel_number}.png'))




