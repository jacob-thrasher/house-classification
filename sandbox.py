import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
import numpy as np
import gc
import pandas as pd
import timm
from tqdm import tqdm
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from transformers import SamModel, SamProcessor, pipeline
from data import ErieParcels, ErieParcels_top4s
from sklearn.model_selection import KFold

def get_data_from_AL_split(root, AL_path, AL_iter, k=200):
    base_df = pd.read_csv(os.path.join(root, 'active_learning/active_learning/train.csv'))
    final_df = pd.read_csv(os.path.join(AL_path, 'train.csv'))
    num_transfered =  k*AL_iter
    num_total = num_transfered + len(base_df)

    trimmed_df = final_df[:num_total]
    valid_df = pd.read_csv(os.path.join(AL_path, 'train.csv'))

    valid_df = valid_df[~valid_df['parcel_number'].isin(trimmed_df['parcel_number'])]

    return trimmed_df, valid_df

root = '/users/jdt0025/scratch/Erie/CV_AL-entropy'
for i in range(0, 5):
    train = pd.read_csv(os.path.join(root, str(i), 'train.csv'))
    test = pd.read_csv(os.path.join(root, str(i), 'test.csv'))

    print(len(train), len(test))


# root = '/users/jdt0025/scratch/Erie'
# al_method = 'entropy'
# al_iter = 45

# train_df, _ = get_data_from_AL_split(root, f'figures/AL-{al_method}', AL_iter=al_iter, k=200)
# test_df = pd.read_csv(os.path.join(root, 'active_learning/active_learning/test.csv'))

# merged_df = pd.concat([train_df, test_df])

# X = merged_df['parcel_number'].tolist()
# y = merged_df['homestead_status'].tolist()

# skf = KFold(n_splits=5, shuffle=True)
# for i, (train_index, test_index) in enumerate(skf.split(X, y)):
#     k_train = merged_df[merged_df.index.isin(train_index)]
#     k_test = merged_df[merged_df.index.isin(test_index)]

#     k_train.to_csv(os.path.join(root, f'CV_AL-{al_method}', str(i), 'train.csv'))
#     k_test.to_csv(os.path.join(root, f'CV_AL-{al_method}', str(i), 'test.csv'))


# root = 'D:\\Big_Data\\Erie'
# dst = 'D:\\Big_Data\\Erie\\cv2'

# df1 = pd.read_csv(os.path.join(root, 'cv/0/train.csv'))
# df2 = pd.read_csv(os.path.join(root, 'cv/0/test.csv'))
# df = pd.concat([df1, df2])

# n_train = int(len(df) * 0.1)
# n_test = int(len(df) * 0.2)

# train = df.groupby('homestead_status', group_keys=False).apply(lambda x: x.sample(frac=0.1))
# df = df[~df['parcel_number'].isin(train['parcel_number'])]

# test = df.groupby('homestead_status', group_keys=False).apply(lambda x: x.sample(frac=0.2))

# valid = df[~df['parcel_number'].isin(test['parcel_number'])]

# print(len(train), len(valid), len(test))


# train.to_csv(os.path.join(root, 'active_learning/train.csv'))
# valid.to_csv(os.path.join(root, 'active_learning/valid.csv'))
# test.to_csv(os.path.join(root, 'active_learning/test.csv'))

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
