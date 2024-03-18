from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
from torch import nn
import timm
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
import cv2
import os
from torchvision import models
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
import torchvision.transforms as T
from data import ErieParcels
from network import train, test_step
import pandas as pd
from utils import create_confusion_matix
import transformers
import random


class ResNetAT(ResNet):
    """Attention maps of ResNeXt-101 32x8d.

    Overloaded ResNet model to return attention maps.
    """

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        g0 = self.layer1(x)
        g1 = self.layer2(g0)
        g2 = self.layer3(g1)
        g3 = self.layer4(g2)

        return [g.pow(2).mean(1) for g in (g0, g1, g2, g3)]

def resnet18():
    # base = models.resnet18(pretrained=True)
    base = models.resnet18(pretrained=True)
    base.fc = nn.Linear(512, 2)

    base.load_state_dict(torch.load('models/BCE_homestead_sgd1.pt'), strict=False)

    model = ResNetAT(BasicBlock, [2, 2, 2, 2])
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(base.state_dict(), strict=False)
    return model

def plot_attention(model,
                    img, out_dir: str,
                    title: str):

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))
    fig.suptitle(title)



    model.eval()
    with torch.no_grad():
        gs = model(img)

    ax[0].imshow(img.squeeze().permute(1, 2, 0))

    for i, g in enumerate(gs):
        ax[i+1].imshow(g[0], interpolation='bicubic', cmap='gray')
        ax[i+1].set_title(f'g{i}')

    plt.show()

    # Save pdf versions
    # Path(f"{out_dir}_pdf").mkdir(parents=True, exist_ok=True)
    # fig_filename = os.path.join(f"{out_dir}_pdf", f"{title}.pdf")
    # fig.savefig(fig_filename, bbox_inches='tight')

    # # Save png versions
    # Path(f"{out_dir}_png").mkdir(parents=True, exist_ok=True)
    # fig_filename = os.path.join(f"{out_dir}_png", f"{title}.png")
    # fig.savefig(fig_filename, bbox_inches='tight')



torch.manual_seed(69)   
np.random.seed(69)
random.seed(69)


root = 'D:\Big_Data'

train_dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erietrain.csv'), year_regression=False)
val_dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erieval.csv'), year_regression=False)


train(train_dataset, val_dataset, model_name='hs_swin_adam3e-5_wd0.01')


# test_dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erietest.csv'), year_regression=False)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# model.fc = nn.Linear(512, 2)
# model.load_state_dict(torch.load('models/BCE_homestead_sgd.pt'), strict=False)
# model.to('cuda')

# loss, acc, f1 = test_step(model, test_dataloader, nn.CrossEntropyLoss(), 'cuda')
# print(acc, f1)

# create_confusion_matix(model, test_dataloader, 'cuda')

# print(len(train_dataset))
# after = 0
# for _, label in train_dataset:
#     after += label

# print(after, len(train_dataset) - after)





# model = resnet18()
# model.eval()

# base = models.resnet18(pretrained=True)
# base.fc = nn.Linear(512, 2)
# base.load_state_dict(torch.load('models/BCE_homestead_sgd1.pt'), strict=False)

# total_positive = 0
# for i in range(len(val_dataset)):
#     img, year = val_dataset[i]

#     pred = base(img.unsqueeze(0))
#     pred = torch.nn.functional.softmax(pred, dim=1)
#     total_positive += torch.argmax(pred, dim=1).item()
#     # print(f'Real: {year}\nPred: {torch.argmax(pred, dim=1)}')
#     # plot_attention(model, img.unsqueeze(0), None, 'Test')
#     # plt.close()

# print(total_positive)


# NOTES

# Regression_1 is best model with Adam lr=0.001






