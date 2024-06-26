import os
import torch
import timm
from torch import nn
from PIL import Image
from collections import OrderedDict
from torch.utils.data import DataLoader
from data import ErieParcels
from tqdm import tqdm
from train_test import test_step

# models: 
# VGG16            : 'vgg16.tv_in1k'
# ResNet50         : 'resnet50.a1_in1k'
# DenseNet121      : 'densenet121.tv_in1k'
# ViT              : 'vit_base_patch16_224'
# SwinTransformerV1: 'swin_large_patch4_window7_224.ms_in22k'
# SwinTransformerV2: 'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k'

root = 'D:\\Big_Data\\Erie'
model_path = 'vgg16.tv_in1k'
pretrained_model_path = 'cv_0/VGG16/best_model.pt' #os.path.join(root, 'best_model.pt')

val_dataset = ErieParcels(os.path.join(root, 'parcels_cleaned'), os.path.join(root, 'cv/0/test.csv'), return_id=True)
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
elif model_path == 'vit_base_patch16_224':
    model.head = nn.Linear(768, 2)
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
model.to('cuda')
model.eval()


val_dataset = ErieParcels(os.path.join(root, 'parcels_cleaned'), os.path.join(root, 'erieval.csv'), year_regression=False)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False) 

_, acc, f1, prec, recall = test_step(model, val_dataloader, nn.CrossEntropyLoss(), 'cuda', show_progress=True)

print("Accuracy : ", acc.item())
print("Precision: ", prec.item())
print("Recall   : ", recall.item())
print("F1 score : ", f1.item())


# out = {
#     'TP': [],
#     'TN': [],
#     'FP': [],
#     'FN': []
# }
# for X, y, _id in tqdm(val_dataset):
#     X = X.to('cuda')
#     pred = model(X.unsqueeze(0)).squeeze()

#     pred = torch.argmax(torch.nn.functional.softmax(pred.cpu(), dim=0)).item()

#     if pred == 1 and y == 1:
#         out['TP'].append(_id)   
#     elif pred == 0 and y == 0:
#         out['TN'].append(_id)   
#     elif pred == 1 and y == 0:
#         out['FP'].append(_id)   
#     elif pred == 0 and y == 1:
#         out['FN'].append(_id)   



# for key in tqdm(out):
#     os.mkdir(os.path.join(root, key))
#     for parcel in out[key]:
#         img = Image.open(os.path.join(root, 'parcels', parcel, '0.png'))
#         img.save(os.path.join(root, key, f'{parcel}.png'))