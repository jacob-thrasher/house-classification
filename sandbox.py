import json
from time import sleep
import pandas as pd
from data import ErieParcels
import matplotlib.pyplot as plt

csvpath = '/home/jacob/Documents/data/parcel_info.csv'


dataset = ErieParcels(dataroot='/home/jacob/Documents/data/parcels', csvpath=csvpath)

img, label = dataset[5000]

print(label)
plt.imshow(img.permute(1, 2, 0))
plt.savefig('test.png')
