import os
import torch
import timm
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from PIL import Image
from collections import OrderedDict
from torch.utils.data import DataLoader
from data import ErieParcels
from tqdm import tqdm
from train_test import test_step
from utils import create_confusion_matix

torch.manual_seed(69)

# models: 
# VGG16            : 'vgg16.tv_in1k'
# ResNet50         : 'resnet50.a1_in1k'
# DenseNet121      : 'densenet121.tv_in1k'
# ViT              : 'vit_base_patch16_224'
# SwinTransformerV1: 'swin_large_patch4_window7_224.ms_in22k'
# SwinTransformerV2: 'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k'


al_method = 'least'
al_iter = 27

fold_accs = []
fold_precs = []
fold_recalls = []
fold_f1s = []
for i in range(0, 4):
    root = '/users/jdt0025/scratch/Erie'
    model_path = 'vit_base_patch16_224'
    model_checkpoint = '/users/jdt0025/timm_models/vit.pt'
    dst = f'figures/AL-{al_method}{str(al_iter)}_folds/{str(i)}'
    pretrained_model_path = os.path.join(dst, 'best_model.pt') #os.path.join(root, 'best_model.pt')

    val_dataset = ErieParcels(os.path.join(root, 'parcels_cleaned'), os.path.join(root, f'CV_AL-{al_method}/{str(i)}/test.csv'), return_id=False, split=None)
    total_images = len(val_dataset)  

    model = timm.create_model(model_path, checkpoint_path=model_checkpoint)

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


    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True) 

    _, acc, f1, prec, recall = test_step(model, val_dataloader, nn.CrossEntropyLoss(), 'cuda', show_progress=True)

    fold_accs.append(acc)
    fold_precs.append(prec)
    fold_recalls.append(recall)
    fold_f1s.append(f1)

    

    print(f'Fold: {i}')
    print("  Accuracy : ", acc)
    print("  Precision: ", prec)
    print("  Recall   : ", recall)
    print("  F1 score : ", f1)


bar_width = 0.2
br1 = np.arange(len(fold_accs))
br2 = [x + bar_width for x in br1]
br3 = [x + bar_width for x in br2]
br4 = [x + bar_width for x in br3]

plt.bar(br1, fold_accs, color='blue', width=bar_width, label='Accuracy')
plt.bar(br2, fold_precs, color='orange', width=bar_width, label='Precision')
plt.bar(br3, fold_recalls, color='lime', width=bar_width, label='Recall')
plt.bar(br4, fold_f1s, color='red', width=bar_width, label='F1')
plt.title(f"Cross Validation for {al_method}:{str(al_iter)}")
plt.legend()
plt.xlabel("Fold")
plt.ylabel("Value")
plt.savefig(f'figures/AL-{al_method}{str(al_iter)}_folds/eval.png')
plt.show()
    # create_confusion_matix(model, val_dataloader, 'cuda', dst=dst)


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

    # for key in out:
    #     print(key, len(out[key]))



# for key in tqdm(out):
#     os.mkdir(os.path.join(root, key))
#     for parcel in out[key]:
#         img = Image.open(os.path.join(root, 'parcels', parcel, '0.png'))
#         img.save(os.path.join(root, key, f'{parcel}.png'))