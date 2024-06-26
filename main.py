import os
import numpy as np
import torch
import random
import transformers
import pytorch_warmup as warmup
import timm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from torch.optim import Adam, SGD, AdamW
from torch import nn
from data import ErieParcels
from train_test import train, test_step
from utils import create_confusion_matix

seed = 99

torch.manual_seed(seed)   
np.random.seed(seed)
random.seed(seed)


# models: 
# VGG16            : 'vgg16.tv_in1k'
# ResNet50         : 'resnet50.a1_in1k'
# DenseNet121      : 'densenet121.tv_in1k'
# ViT              : 'vit_base_patch16_224'
# SwinTransformerV1: 'swin_large_patch4_window7_224.ms_in22k'
# SwinTransformerV2: 'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k'

# folds = 5
# for i in range(folds):

models = ['vgg16.tv_in1k', 'resnet50.a1_in1k', 'vit_base_patch16_224', 'swin_large_patch4_window7_224.ms_in22k']
names = ['VGG16', 'ResNet50', 'ViT', 'SwinV1']
for m, n in zip(models, names):
    root = 'D:\\Big_Data\\Erie'

    config = {
        'dst': 'cv_0',
        'model_name': n,
        'model_path': m,
        'epochs': 5,
        'batch_size': 64,
        'accum_iter': 1,
        'warmup_epochs': 1,
        'lr': 3e-5,
        'weight_decay': 0,
        'schedulers': None,
        'show_progress': True
    }

    # model = transformers.Swinv2ForImageClassification.from_pretrained(config['model_path'])
    # model = transformers.ViTForImageClassification.from_pretrained(config['model_path'])
    model = timm.create_model(config['model_path'], pretrained=True)

    print(model)

    if config['model_path'] == 'vgg16.tv_in1k':
        model.head.fc = nn.Sequential(nn.Linear(4096, 1000), 
                                    nn.ReLU(),
                                    nn.Linear(1000, 2))
    elif config['model_path'] == 'resnet50.a1_in1k':
        model.fc = nn.Sequential(nn.Linear(2048, 1000), 
                                nn.ReLU(),
                                nn.Linear(1000, 2))
    elif config['model_path'] == 'vit_base_patch16_224':
        model.head = nn.Linear(768, 2)
    elif config['model_path'] == 'swin_large_patch4_window7_224.ms_in22k':
        model.head.fc = nn.Linear(1536, 2)

    optim = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # train_dataset = ErieParcels(os.path.join(root, 'parcels_cleaned'), os.path.join(root, 'cv', str(i), 'train.csv'), year_regression=False)
    # val_dataset = ErieParcels(os.path.join(root, 'parcels_cleaned'), os.path.join(root, 'cv', str(i), 'test.csv'), year_regression=False)

    train_dataset = ErieParcels(os.path.join(root, 'parcels_cleaned'), os.path.join(root, 'cv/0/train.csv'), year_regression=False)
    val_dataset = ErieParcels(os.path.join(root, 'parcels_cleaned'), os.path.join(root, 'cv/0/test.csv'), year_regression=False)

    print(len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False) 


    # iter_per_epoch = int(len(train_dataloader) / config['accum_iter'])
    # warmup_period = config['warmup_epochs'] * iter_per_epoch
    # warmup_scheduler = warmup.LinearWarmup(optim, warmup_period=warmup_period)
    # cosine_scheduler = CosineAnnealingLR(optim, T_max=iter_per_epoch*(config['epochs'] - config['warmup_epochs']))

    print("Len dataloader", len(train_dataloader))
    # print("Iterations per epoch", iter_per_epoch)

    # schedulers = {
    #     'warmup': warmup_scheduler,
    #     'warmup_period': warmup_period,
    #     'lr_scheduler': cosine_scheduler
    # }
    # config['schedulers'] = schedulers






    metrics = train(train_dataloader, val_dataloader, model, optim, config)

    create_confusion_matix(model, val_dataloader, 'cuda', dst=os.path.join(config['dst'], config['model_name']))
    # print(f"CV {i}:\n\tAcc: {acc}\n\t F1: {f1}")
    print(metrics)
