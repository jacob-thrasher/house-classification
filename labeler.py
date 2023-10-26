import os
import csv
from random import shuffle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


root = 'D:\\Big_Data\\zillow_images'
files = os.listdir(root)
shuffle(files)

out = open('labels.csv', 'w', newline='')
writer = csv.writer(out)
header = ['file', 'label']
writer.writerow(header)

label = -1
for file in files:
    img = Image.open(os.path.join(root, file))
    plt.imshow(np.asarray(img))
    plt.show()

    while label not in [0, 1, 5, 9]:
        label = int(input('Label: '))

    if label == 5: continue
    if label == 9: break
    writer.writerow([file, label])

    label = -1

out.close()
