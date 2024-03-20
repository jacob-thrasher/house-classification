import os
import csv
from random import shuffle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from train_test import CNN
import torch
from data import ZillowUnsupervised

def manual_label():
    plt.ion()
    root = 'D:\\Big_Data\\zillow_images'
    files = os.listdir(root)
    shuffle(files)

    out = open('labels.csv', 'a', newline='')
    writer = csv.writer(out)
    header = ['file', 'label']
    writer.writerow(header)

    label = -1
    tot_outdoor = 0
    tot_indoor = 0
    for file in files:
        img = Image.open(os.path.join(root, file))
        plt.imshow(np.asarray(img))
        plt.show()

        while label not in [0, 1, 5, 9]:
            label = int(input('Label: '))

        

        if label == 0: tot_outdoor += 1
        elif label == 1: tot_indoor += 1
        elif label == 5: 
            label = -1
            img.close()
            plt.close()
            continue
        elif label == 9: break

        print('Total outdoor:', tot_outdoor, '\nTotal indoor :', tot_indoor)

        writer.writerow([file, label])

        label = -1
        img.close()
        plt.close()

    out.close()

def automatic_labeler(path_to_state_dict, dataroot):
    f = open('exteriors.txt', 'w')

    model = CNN()
    model.load_state_dict(torch.load(path_to_state_dict))
    model.eval()
    
    dataset = ZillowUnsupervised(dataroot)

    for img, filename in dataset:
        img = img.unsqueeze(0)
        pred = model(img)
        label = np.round(pred.item())

        if int(label) == 1:
            f.write(f'{filename}\n')
        

# automatic_labeler('CNN.pt', 'D:\\Big_Data\\zillow_images')

root = 'D:\\Big_Data\\zillow_test'

with open('exteriors.txt', 'r') as f:
    files = f.readlines()

preds = [('.').join(x.split('.')[:-1]) for x in files]

rootfiles = os.listdir(root)

files = [i for i in rootfiles if i not in preds]



for file in files:
    path = os.path.join(root, file)
    os.remove(path)
