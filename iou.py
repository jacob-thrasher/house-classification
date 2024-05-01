import os
import numpy as np
import torch
import timm
import matplotlib.pyplot as plt
import pprint
from collections import OrderedDict
from tqdm import tqdm
from PIL import Image
from torchmetrics.classification import BinaryJaccardIndex
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from data import ErieParcels

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def load_mask(mask_path):
    img = Image.open(mask_path).convert('L')
    img = np.round((np.array(img) / 255) * 20)
    img = np.reshape(img, (224, 224)) # PROBABLY UNSAFE
    return img

def load_batch_mask(root, mask_ids):
    masks = []
    for _id in mask_ids:
        m = load_mask(os.path.join(root, f'{_id}.png'))
        masks.append(m)

    
    return masks

root = '/home/jacob/Documents/data/erie_data'
val_masks = os.path.join(root, 'masksval_224')
pretrained_model_path = 'best_model.pt'

categories = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 
            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


val_dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erieval.csv'), year_regression=False, model_path=None, return_id=True)
dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)
total_images = len(val_dataset)  

model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.head = torch.nn.Linear(768, 2)

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

cam = GradCAM(model=model, target_layers=[model.blocks[-1].norm1], reshape_transform=reshape_transform)
target = None

scores = {}
for c in categories: scores[c] = 0

for imgs, _, parcel_numbers in tqdm(dataloader):
    imgs = imgs.to('cuda')

    grayscale_cam = cam(input_tensor=imgs, targets=target)

    imgs = imgs.cpu()
    masks = load_batch_mask(val_masks, parcel_numbers)


    for mask, attn, img in zip(masks, grayscale_cam, imgs):
        # Check if house in mask
        unique, counts = np.unique(mask, return_counts=True)
        cat_counts = dict(zip(unique, counts))
        if 3 not in list(cat_counts.keys()): # Building
            total_images -= 1
            continue # Skip if no buildling

        # print(np.min(mask))
        # plt.imshow(mask)
        # plt.savefig('test_imgs/mask.png')
        # plt.close()

        for i in range(len(categories)):
            category = categories[i]
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

            scores[category] += normalized_sum

            # plt.imshow(img.permute(1, 2, 0))
            # plt.imshow(attn*category_mask, cmap='jet', alpha=0.7)
            # plt.savefig(f'test_imgs/{category}.png')
            # plt.close()

# Average scores
for key in scores: scores[key] /= total_images
total_skipped = len(val_dataset) - total_images
print("Total images skipped:", total_skipped)

scores = dict(sorted(scores.items(), key=lambda item: item[1]))
pprint.pp(scores)