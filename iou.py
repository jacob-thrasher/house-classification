import os
import numpy as np
import torch
import timm
from collections import OrderedDict
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
val_masks = os.path.join(root, 'masksval')
pretrained_model_path = 'best_model.pt'

categories = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 
            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


val_dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erieval.csv'), year_regression=False, model_path=None, return_id=True)
dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)  

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

imgs, _, parcel_numbers = next(iter(dataloader))
imgs = imgs.to('cuda')
model.to('cuda')

cam = GradCAM(model=model, target_layers=[model.blocks[-1].norm1], reshape_transform=reshape_transform)
target = None
grayscale_cam = cam(input_tensor=imgs, targets=target)

imgs = imgs.cpu()

threshold = .5
pred = grayscale_cam[1, :]
print(pred.shape)

matric = BinaryJaccardIndex(threshold=0.5)

masks = load_batch_mask(val_masks, parcel_numbers)

scores = {}
for i in range(len(categories)):
    category = categories[i]
    
    # Build category mask
    category_masks = []
    for m in masks:
        print(m.shape)
        m = m[m == i]
        print(m.shape)
        
    break