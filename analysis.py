import os
import torch
import transformers
import timm
import matplotlib.pyplot as plt
import numpy as np
import torchmetrics.functional as tmf
import csv

from collections import OrderedDict
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from data import ErieParcels, ErieParcels_top4s
from tqdm import tqdm
from utils import create_confusion_matix, load_model, load_batch_mask, get_sorted_scores

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

split = None
k = 4
threshold = 0

root = '/users/jdt0025/scratch/Erie'
val_masks = os.path.join(root, 'active_learning/masks')
model_path = 'vit_base_patch16_224'
finetuned_model_path = 'figures/AL-entropy/iter_45/best_model.pt'
pretrained_model_checkpoint = '/users/jdt0025/timm_models/vit.pt'

static_categories = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky']

test_dataset = ErieParcels_top4s(os.path.join(root, 'parcels_cleaned'), os.path.join(root, 'active_learning/active_learning/top-4s.csv'), split=split)
dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)
print(len(test_dataset))

model, target_layers = load_model(model_path, pretrained_model_checkpoint, finetuned_model_path, get_gradcam_layers=True)
device = 'cuda'
model.to(device)


cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) # Reshape transform for ViT
target = None


f = open('semantic_rankings.csv', 'w')
writer = csv.writer(f)
header = ['ID', 'label', 'isCorrect', 'top1_rank', 'top2_rank', 'top3_rank', 'top4_rank']
writer.writerow(header)


model.eval()
acc = 0
f1 = 0
prec = 0
recall = 0
counts = torch.tensor([0, 0])
stat_counts = torch.tensor([0, 0, 0, 0, 0])
for imgs, info_batch in tqdm(dataloader, disable=False):
    imgs = imgs.to('cuda')

    # Get attention and load masks
    grayscale_cam = cam(input_tensor=imgs, targets=target)
    masks = load_batch_mask(val_masks, info_batch['parcel_number'])
    out = model(imgs)
    pred = torch.argmax(torch.nn.functional.softmax(out.cpu(), dim=0)).item()

    dict_list = []
    for i in range(len(info_batch)):
        this_dict = {
            'homestead_status': info_batch['homestead_status'][i],
            'parcel_number': info_batch['parcel_number'][i],
            'prediction': pred[i],
            'top4': [x[i] for x in info_batch['top4']]
        }
        dict_list.append(this_dict)


    imgs = imgs.cpu()
    for mask, attn, img, info in zip(masks, grayscale_cam, imgs, dict_list):

        # Score for this image
        image_scores = {}

        for i in range(len(static_categories)):
            category = static_categories[i]
            category_mask = mask.copy()

            category_mask = np.where(category_mask == (i+1), category_mask, 0)
            category_mask /= (i+1) # Set values to 1

            weighted_sum = np.sum(attn * category_mask)
            total_nonzero = np.count_nonzero(category_mask)

            if total_nonzero > 0:
                normalized_sum = weighted_sum / total_nonzero
            else: normalized_sum = 0

            image_scores[category] = normalized_sum

        relevent_items = list(info['top4'])
        sorted_items, sorted_values = get_sorted_scores(image_scores)

        print(sorted_values)    
        break
        predicted_ranks = [sorted_items.index(x) for x in relevent_items]
        # ['ID', 'label', 'isCorrect', 'top1_rank', 'top2_rank', 'top3_rank', 'top4_rank']
        row = [info['parcel_number'], info['homestead_status'], info['homestead_status'] == info['prediction']]
        row += predicted_ranks
        writer.writerow(row)




