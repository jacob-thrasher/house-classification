import torchmetrics
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm

def plot_confusion_matrix(pred, labels, classes):
    cm = confusion_matrix(labels, pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix on test set',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax

def create_confusion_matix(model, dataloader, device):
    print("Generating confusion matrix...")
    all_pred = []
    all_labels = []

    model.eval()
    for i, (X, y) in enumerate(tqdm(dataloader)):
        X = X.to(device)
    
        pred = model(X)
        pred = torch.round(pred.detach().cpu()).squeeze()
        pred = [int(x.item()) for x in list(pred)]
        all_pred += pred
        all_labels += y.tolist()

    fig, ax = plot_confusion_matrix(all_pred, all_labels, classes=['Active', 'Inactive'])
    plt.show()
    fig.savefig('metrics\\cm.png')

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

