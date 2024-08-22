import os
import numpy as np
import pandas as pd
import torchmetrics
import torch
import matplotlib.pyplot as plt
import torchmetrics.functional as tmf
import timm
from torch import nn
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict

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


#########################
# ACTIVE LEARNING UTILS #
#########################

def compute_uncertainty_scores(logits, method, temperature=1):
    '''
    Computes uncertainty scores, where higher values indicate higher uncertainty

    Args:
        logits: Model outputs (no softmax)
        method: Uncertainty calculation method, where:
                'least' = least confidence 
                'entropy' = entropy sampling
    '''
    assert method in ['least', 'entropy', 'margin', 'ratio'], f'Expexted parameter method to be in [least, entropy, margin, ratio], got {method}'

    # Apply softmax function
    probs = nn.functional.softmax(logits, dim=1)
    n_classes = probs.size()[1]
    max_values = torch.max(probs, dim=1)
    pred_class = max_values.indices

    if method == 'least': # Least confidence
        max_probs = max_values.values
        # s = (1 - max_probs) * (n_classes / (n_classes-1))
        s = 1 - max_probs

    elif method == 'entropy':
        s = -torch.sum(probs * torch.log2(probs), dim=1) / torch.log2(torch.tensor(n_classes))

    elif method == 'margin':
        top2 = torch.topk(probs, 2, dim=1).values
        s = 1 - (top2[:, 0] - top2[:, 1])

    elif method == 'ratio':
        top2 = torch.topk(probs, 2, dim=1).values
        s = -(top2[:, 0] / top2[:, 1])

    return s.tolist(), pred_class.tolist()

def undersample(path, pids, col='ID'):
    '''
    Undersamples datasets by transferring pids from validation set to training set

    Args:
        path: path to load/save csvs
        pids: list of pids to transfer
    
    Keyword Args:
        col: column to sample from
    '''
    al_train = pd.read_csv(os.path.join(path, 'train.csv'))
    al_valid = pd.read_csv(os.path.join(path, 'valid.csv'))

    transfer_rows = al_valid[al_valid[col].isin(pids)]
    al_train = pd.concat([al_train, transfer_rows], ignore_index=True) # Move rows to train set
    al_valid.drop(index=transfer_rows.index, inplace=True)

    assert len(set(al_train[col].tolist()).intersection(set(al_valid[col].tolist()))) == 0, f'Found overlap in train and validation set!!'

    al_train.to_csv(os.path.join(path, 'train.csv'), index=False)
    al_valid.to_csv(os.path.join(path, 'valid.csv'), index=False)

def update_splits(data_path, valid_dataloader, model_path, n_transfer=2, device='cuda', uncertainty='least'):
    '''
    Performs uncertainty calculations and updates train/validation csvs with most difficult subjects

    Args:
        validpath: path to validation dataset
        model_path: path to model TODO: Refactor to infer model ppath from valid_path
        n_transfer: number of subjects to transfer
        device: operating device
        uncertainty: method for computing uncertainty
    '''

    print('\nUpdating splits....')
    # Load best model
    model = timm.create_model('vit_base_patch16_224', checkpoint_path='/users/jdt0025/timm_models/vit.pt')
    model.head = nn.Linear(768, 2)

    state_dict = torch.load(model_path)
    new_dict = OrderedDict()
    for key in state_dict:
        value = state_dict[key]
        if 'module' in key:
            key = key.replace('module.', '')

        new_dict[key] = value

    model.load_state_dict(new_dict)
    model.to(device)
    model.eval()

    # scaled_model = ModelWithTemperature(model)
    # scaled_model.set_temperature(test_dataloader)

    
    uncertainty_df = pd.DataFrame(columns=['ID', 'label', 'Pred', 'Uncertainty'])
    for X, y, pid in tqdm(valid_dataloader, disable=False):
        X = X.to(device)
        y = y.tolist()
        pid = pid

        out = model(X)

        scores, preds = compute_uncertainty_scores(out, method=uncertainty, temperature=1)

        # TODO: Get rid of this for loop
        for p, s, _id, label in zip(preds, scores, pid, y):
            uncertainty_df.loc[len(uncertainty_df)] = [_id, label, p, s]


    uncertainty_df.sort_values(by='Uncertainty', ascending=False, inplace=True)

    n_transfer = min(n_transfer, len(uncertainty_df))

    undersample(data_path, uncertainty_df['ID'][:n_transfer].tolist(), col='parcel_number')