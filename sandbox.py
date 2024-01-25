import json
from time import sleep
import pandas as pd
from data import ErieParcels
import matplotlib.pyplot as plt

csvpath = 'D:\\Big_Data\\parcel_info.csv'


dataset = ErieParcels(dataroot='D:\\Big_Data\\parcels', csvpath=csvpath)


# for i in range(5):
#     img, label = dataset[i]

#     print(label)
#     plt.imshow(img.permute(1, 2, 0))
#     plt.show()