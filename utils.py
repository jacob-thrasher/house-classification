import os
import numpy as np
import pandas as pd
import torchmetrics
import torch
import matplotlib.pyplot as plt
import torchmetrics.functional as tmf
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from PIL import Image

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

def create_confusion_matix(model, dataloader, device, dst):
    print("Generating confusion matrix...")
    all_pred = []
    all_labels = []

    model.eval()
    for i, (X, y) in enumerate(tqdm(dataloader)):
        X = X.to(device)
    
        pred = model(X)
        # pred = torch.round(pred.detach().cpu()).squeeze()
        pred = torch.argmax(pred, dim=1)
        pred = [int(x.item()) for x in list(pred)]
        all_pred += pred
        all_labels += y.tolist()

    fig, ax = plot_confusion_matrix(all_pred, all_labels, classes=['Active', 'Inactive'])
    fig.savefig(os.path.join(dst, 'cm.png'))

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

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def load_mask(mask_path):
    img = Image.open(mask_path).convert('L')
    img = img.resize((224, 224))
    test = np.array(img)
    img = np.round((np.array(img) / 255) * 20)
    img = np.reshape(img, (224, 224)) # PROBABLY UNSAFE
    return img

def load_batch_mask(root, mask_ids):
    masks = []
    for _id in mask_ids:
        m = load_mask(os.path.join(root, f'{_id}.png'))
        masks.append(m)

    
    return masks

def get_sorted_scores(scores):
    scores = dict(sorted(scores.items(), key=lambda item: item[1]))
    ranked_items = list(scores.keys())
    ranked_items.reverse()
    ranked_values = list(scores.values())
    ranked_values.reverse()

    return ranked_items, ranked_values

def precision_at_k(scores, k, relevant_items):
    ranked_items, _ = get_sorted_scores(scores)
    top_k = ranked_items[:k]

    matches = set(top_k).intersection(set(relevant_items))

    return len(matches) / k

# def mean_average_precision(scores, k, relevant_items, threshold=0.001):
#     ranked_items, ranked_values = get_sorted_scores(scores)
#     threshold_values = [x for x in ranked_values if x >= threshold]

#     if len(threshold_values) < k: k = len(threshold_values)
#     top_k = ranked_items[:k]

#     all_precisions = []
#     for i, item in enumerate(top_k):
#         if item in relevant_items:
#             precision = precision_at_k(scores, i+1, relevant_items)
#             all_precisions.append(precision)

#     if len(all_precisions) > 0:
#         return sum(all_precisions) / len(all_precisions)
#     return 0

def get_retrieval_metrics(scores, relevent_items, k=4, threshold=0):
    ranked_items, ranked_values = get_sorted_scores(scores)
    ranked_values = torch.tensor(ranked_values)
    target_indicators = torch.tensor([x in relevent_items for x in ranked_items])

    mAP = tmf.retrieval_average_precision(torch.nn.functional.softmax(ranked_values, dim=0), target_indicators, top_k=k)
    ndcg = tmf.retrieval.retrieval_normalized_dcg(ranked_values, target_indicators, top_k=k)
    rprec = tmf.retrieval.retrieval_r_precision(torch.nn.functional.softmax(ranked_values, dim=0)[:k], target_indicators[:k])

    return mAP, ndcg, rprec

def mean_average_precision(scores, relevent_items, k=4, threshold=0):
    ranked_items, ranked_values = get_sorted_scores(scores)
    target_indicators = [x in relevent_items for x in ranked_items]
    return tmf.retrieval_average_precision(torch.nn.functional.softmax(torch.tensor(ranked_values), dim=0), torch.tensor(target_indicators), top_k=k)

def NDCG(scores, relevent_items, k=4, threshold=0):
    ranked_items, ranked_values = get_sorted_scores(scores)
    target_indicators = [x in relevent_items for x in ranked_items] 

    for i in range(len(ranked_values)):
        if ranked_values[i] < threshold: ranked_values[i] = 0

    return tmf.retrieval.retrieval_normalized_dcg(torch.tensor(ranked_values), torch.tensor(target_indicators), top_k=k)

