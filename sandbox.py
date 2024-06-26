import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
import numpy as np
import gc
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from transformers import SamModel, SamProcessor, pipeline
from data import ErieParcels, ErieParcels_top4s


root = 'D:\\Big_Data\\Erie'
dst = 'D:\\Big_Data\\Erie\\parcels_cleaned'

# df1 = pd.read_csv(os.path.join(root, 'cv/1/train.csv'))
# df2 = pd.read_csv(os.path.join(root, 'cv/1/test.csv'))
# concat = pd.concat([df1, df2])

test = pd.read_csv(os.path.join(root, 'cv/0/test.csv'))
top4_df = pd.read_csv(os.path.join(root, 'top-4s_val.csv'))

# print(len(cv_df))
# print(len(top4_df))
top4_df = top4_df[top4_df['parcel_number'].isin(test['parcel_number'])]
top4_df.to_csv(os.path.join(root, 'cv/0/top-4s.csv'), index=False)







# generator = pipeline('image-segmentation', model="nvidia/segformer-b1-finetuned-cityscapes-1024-1024", device=0)


# categories = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 
#             'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

# for f in tqdm(os.listdir(root)):
#     folder = os.path.join(root, f)
#     if len(folder) < 1: continue

#     most_building = None
#     most_building_perc = 0
#     for i in os.listdir(folder):
#         img = Image.open(os.path.join(folder, i))

#         outputs = generator(img, points_per_batch=64)

#         # Find building mask
#         mask = None
#         for m in outputs:
#             if m['label'] == 'building':
#                 mask = np.array(m['mask'])    
#                 break
#         if mask is None: continue

#         # Confirm building occupies sufficient percentage of image
#         building_perc = np.sum(mask / 255) / (mask.shape[0] * mask.shape[1])
#         if building_perc < 0.05: continue
#         elif building_perc > most_building_perc: 
#             most_building_perc = building_perc
#             most_building = i
    
#     if most_building is not None:
#         img = Image.open(os.path.join(folder, most_building))
#     else:
#         img = Image.open(os.path.join(folder, '0.png'))
#     img.save(os.path.join(dst, f'{f}.png'))
