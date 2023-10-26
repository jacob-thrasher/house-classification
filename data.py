import os
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Zillow(Dataset):
    def __init__(self, root, img_dim=224):

        self.preprocess = T.Compose([
            T.Resize(img_dim),
            T.CenterCrop(img_dim),
            T.ToTensor(),
            T.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        self.root = root
        self.files = os.listdir(root)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.files[idx])).convert('RGB')
        return self.preprocess(img), self.files[idx]
    
def cluster(root, n=2):
    dataset = Zillow(root)

    # Temp soln
    imgs = []
    for i in range(0, 1000):
        imgs.append(dataset[i])

    kmeans = KMeans(n_clusters=2, init='random')    
    print("Fitting data")
    kmeans.fit(imgs)

    print("Making predictions")
    Z = kmeans.predict(imgs)

    plot_clusters(Z, imgs)

def plot_clusters(Z, data):
    for i in range(0,2):
        row = np.where(Z==i)[0]  # row in Z for elements of cluster i
        num = row.shape[0]       #  number of elements for each cluster
        r = np.floor(num/10.)    # number of rows in the figure of the cluster 

        print("cluster "+str(i))
        print(str(num)+" elements")

        plt.figure(figsize=(10,10))
        for k in range(0, num):
            plt.subplot(r+1, 10, k+1)
            image = data[row[k], ]
            image = image.reshape(8, 8)
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.show()

cluster('C:\\Users\\jthra\\Documents\\data\\zillow_images_copy\\zillow_images')