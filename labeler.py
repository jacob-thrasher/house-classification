import os
import csv
from random import shuffle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

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
