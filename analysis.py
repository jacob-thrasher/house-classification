import os
import torch
import transformers
import timm
import matplotlib.pyplot as plt
import numpy as np
import torchmetrics.functional as tmf

from collections import OrderedDict
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from data import ErieParcels
from tqdm import tqdm
from utils import create_confusion_matix

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

root = 'D:\\Big_Data\\Erie'
model_path = "google/vit-base-patch16-224"
pretrained_model = 'vit_adam-lr3e-5_no-wd_seed_99/best_model.pt'

test_dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erieval.csv'), year_regression=False, split=None)
dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)
print(len(test_dataset))

# model = transformers.Swinv2ForImageClassification.from_pretrained(model_path)
# model = transformers.ViTForImageClassification.from_pretrained(model_path)
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.head = torch.nn.Linear(768, 2)

state_dict = torch.load(pretrained_model)
new_dict = OrderedDict()
for key in state_dict:
    value = state_dict[key]
    if 'module' in key:
        key = key.replace('module.', '')

    new_dict[key] = value

model.load_state_dict(new_dict)
device = 'cuda'
model.to(device)

model.eval()
acc = 0
f1 = 0
prec = 0
recall = 0
counts = torch.tensor([0, 0])
stat_counts = torch.tensor([0, 0, 0, 0, 0])
for (X, y) in tqdm(dataloader, disable=True):
    X = X.to(device)
    # y = y.type(torch.float32).to(device)
    # out = model(X).logits
    out = model(X)

    prediction = torch.argmax(out, dim=1).detach().cpu()
    counts += torch.bincount(prediction, minlength=2)
    acc += tmf.classification.accuracy(prediction, y, task='binary')
    f1 += tmf.f1_score(prediction, y, task='binary')
    prec += tmf.precision(prediction, y, task='binary')
    recall += tmf.recall(prediction, y, task='binary')
    stat_counts += tmf.stat_scores(prediction, y, task='binary')

print(' Acc:', (acc / len(dataloader)).item())
print('  F1:', (f1 / len(dataloader)).item())
print('Prec:', (prec / len(dataloader)).item())
print(' Rec:', (recall / len(dataloader)).item())
print('\nCounts:', counts)
print('Stat counts:', stat_counts)

