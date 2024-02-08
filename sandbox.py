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
from network import train

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
    base.fc = nn.Linear(512, 1)

    base.load_state_dict(torch.load('models/regression.pt'), strict=False)

    model = ResNetAT(BasicBlock, [2, 2, 2, 2])
    model.fc = nn.Linear(512, 1)
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





root = 'D:\Big_Data'
train_dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erietrain.csv'), year_regression=True)
test_dataset = ErieParcels(os.path.join(root, 'parcels'), os.path.join(root, 'erietest.csv'), year_regression=True)

train(train_dataset, test_dataset, model_name='regression')





# model = resnet18()
# model.eval()

# base = models.resnet18(pretrained=True)
# base.fc = nn.Linear(512, 1)
# base.load_state_dict(torch.load('models/regression.pt'), strict=False)
# for i in range(4):
#     img, year = test_dataset[i]

#     pred = base(img.unsqueeze(0))
#     print(f'Real: {year}\nPred: {pred}')
#     plot_attention(model, img.unsqueeze(0), None, 'Test')
#     plt.close()


# NOTES

# Regression_1 is best model with Adam lr=0.001






