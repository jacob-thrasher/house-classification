import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import timm
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam, RMSprop, SGD
from data import ErieParcels
from torch import nn
from torch.utils.data import DataLoader
from utils import update_splits
from train_test import train_step, test_step
from collections import OrderedDict
import transformers
# from temp_scaling import ModelWithTemperature

# models: 
# VGG16            : 'vgg16.tv_in1k'
# ResNet50         : 'resnet50.a1_in1k'
# DenseNet121      : 'densenet121.tv_in1k'
# ViT              : 'vit_base_patch16_224'
# SwinTransformerV1: 'swin_large_patch4_window7_224.ms_in22k'
# SwinTransformerV2: 'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k'

torch.manual_seed(69)

root = '/users/jdt0025/scratch/Erie'
exp_name = 'AL-margin2'
dst = os.path.join('figures', exp_name)
uncertainty = 'margin'
model_path = '/users/jdt0025/timm_models/vit.pt'

if not os.path.exists(dst):
    # Prep files (altered AL csvs are saved for further analysis if necessary)
    os.mkdir(dst)
    shutil.copy(os.path.join(root, 'active_learning/active_learning/train.csv'), os.path.join(dst, f'train.csv'))
    shutil.copy(os.path.join(root, 'active_learning/active_learning/valid.csv'), os.path.join(dst, f'valid.csv'))
    # shutil.copy('figures/AL-margin/train.csv', os.path.join(dst, f'train.csv'))
    # shutil.copy('figures/AL-margin/valid.csv', os.path.join(dst, f'valid.csv'))
else:
    raise OSError(f'Directory {dst} already exists')

dim = 224
al_iter = 25
n_transfer = 200
lr = 3e-5
epochs = 5



al_f1s = []
for i in range(al_iter):
    print(f"\n-------------------------")
    print(f"STARTING AL ITERATION {i}")
    print(f"-------------------------")

    os.mkdir(os.path.join(dst, f'iter_{i}'))



    train_dataset = ErieParcels(os.path.join(root, 'parcels_cleaned'), os.path.join(dst, f'train.csv'), year_regression=False, return_id=True)
    test_dataset = ErieParcels(os.path.join(root, 'parcels_cleaned'), os.path.join(root, f'active_learning/active_learning/test.csv'), year_regression=False, return_id=True)

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    print(len(train_dataset), len(test_dataset))


    model = timm.create_model('vit_base_patch16_224', checkpoint_path=model_path)
    model.head = nn.Linear(768, 2)

    # if model_path == 'vgg16.tv_in1k':
    #     model.head.fc = nn.Sequential(nn.Linear(4096, 1000), 
    #                                 nn.ReLU(),
    #                                 nn.Linear(1000, 2))
    # elif model_path == 'resnet50.a1_in1k':
    #     model.fc = nn.Sequential(nn.Linear(2048, 1000), 
    #                             nn.ReLU(),
    #                             nn.Linear(1000, 2))
    # elif model_path in ['vit_base_patch16_224', '/users/jdt0025/hf_models/vit-base-patch16-224']: # TODO: Fix this obviously
    #     model.head = nn.Linear(768, 2)
    # elif model_path == 'swin_large_patch4_window7_224.ms_in22k':
    #     model.head.fc = nn.Linear(1536, 2)

    device = 'cuda'
    print(f'Using {device}')
    model.to(device)

    loss_fn = CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=lr)

    best_acc = 0
    best_f1 = 0
    train_losses = []
    valid_losses = []
    accs = []
    f1s = []
    for epoch in range(epochs):
        print(f'Epoch: {[epoch]}/{[epochs]}')

        train_loss, _ = train_step(model, train_dataloader, optim, loss_fn, device)
        valid_loss, acc, f1, _, _ = test_step(model, test_dataloader, loss_fn, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        accs.append(acc)    
        f1s.append(f1)

        if f1 >= best_f1:
            best_acc = acc
            best_f1 = f1
            torch.save(model.state_dict(), f'{dst}/iter_{i}/best_model.pt')


        print('Train loss:', train_loss)
        print('Valid loss:', valid_loss)
        print('Accuracy:', acc)
        print('F1:', f1)


        # Plotting
        plt.plot(train_losses, color='blue', label='Train')
        plt.plot(valid_losses, color='orange', label='Valid')
        plt.title('Train and validation loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(f'{dst}/iter_{i}/loss.png')
        plt.close()

        plt.plot(accs, color='green', label='Accuracy')
        plt.plot(f1s, color='purple', label='F1-score')
        plt.title('Metrics over time')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(f'{dst}/iter_{i}/acc_f1.png')
        plt.close()

    # Post training metrics and plotting
    print('-----------')
    print("Best Acc:", best_acc)
    print("Best F1 :", best_f1)
    al_f1s.append(best_f1)

    plt.plot(al_f1s, color='red', label='f1')
    plt.title('Best f1 over active learning iterations')
    plt.xlabel('AL iter')
    plt.legend()
    plt.savefig(f'{dst}/f1s.png')
    plt.close()

    # Active Learning step
    valid_dataset = ErieParcels(os.path.join(root, 'parcels_cleaned'), os.path.join(dst, f'valid.csv'), year_regression=False, return_id=True)

    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    # update_splits_subject(dst, f'{dst}/iter_{i}/best_model.pt', device=device, n_transfer=n_transfer, uncertainty=uncertainty)
    update_splits(dst, valid_dataloader, f'{dst}/iter_{i}/best_model.pt', device=device, n_transfer=n_transfer, uncertainty=uncertainty)