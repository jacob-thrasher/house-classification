import torch
from torch import nn
from torch.optim import Adam, SGD, RMSprop
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import torchmetrics.functional as tmf
import os
from torchvision import models
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.ops import focal_loss
from torch.nn.functional import binary_cross_entropy
from tqdm import tqdm
import transformers

class ResNetAT(ResNet):
    """Attention maps of ResNeXt-101 32x8d.

    Overloaded ResNet model to return attention maps.
    """

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        g0 = self.layer1(x)
        g1 = self.layer2(g0)
        g2 = self.layer3(g1)
        g3 = self.layer4(g2)

        return [g.pow(2).mean(1) for g in (g0, g1, g2, g3)]

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2)
        ])
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(30720, 1)

        self.head = nn.Sequential(nn.Flatten(),
                                nn.Linear(2304, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.dropout(x)
        x = self.head(x)
        return nn.functional.sigmoid(x)
    

def resnet18():
    base = models.resnet18(pretrained=True)
    model = ResNetAT(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(base.state_dict())
    return model

def train_step(model, dataloader, optim, loss_fn, device, show_progress=True, accum_iter=1, schedulers=None):
    model.train()
    epoch_loss = 0
    lrs = []

    for batch, (X, y) in enumerate(tqdm(dataloader, disable=(not show_progress))):
        X = X.to(device)
        # y = y.type(torch.float32).to(device)
        y = y.to(device)

        # out = model(X).logits
        out = model(X)
        # loss = loss_fn(out.logits.squeeze(), y)
        loss = loss_fn(out.squeeze(), y)
        loss.backward()

        epoch_loss += loss.item()
        loss = loss.item() / accum_iter # Normalize loss for accumulation

        # Gradient accumulation
        if (batch + 1) % accum_iter == 0 or (batch + 1) == len(dataloader):
            optim.step()
            optim.zero_grad()

            if schedulers is not None:
                warmup = schedulers['warmup']
                lr_scheduler = schedulers['lr_scheduler']

                with warmup.dampening():
                    if warmup.last_step + 1 >= schedulers['warmup_period']:
                        lr_scheduler.step()
            lrs.append(optim.param_groups[0]['lr'])

    return epoch_loss / len(dataloader), lrs

def test_step(model, dataloader, loss_fn, device, show_progress=True):
    model.eval()
    running_loss = 0
    for (X, y) in tqdm(dataloader, disable=(not show_progress)):
        X = X.to(device)
        # y = y.type(torch.float32).to(device)
        y = y.to(device)
        # out = model(X).logits
        out = model(X)

        loss = loss_fn(out.squeeze(), y)

        running_loss += loss.item()


    prediction = torch.argmax(out, dim=1)
    acc = tmf.classification.accuracy(prediction, y, task='binary').cpu()
    f1 = tmf.f1_score(prediction, y, task='binary').cpu()
    prec = tmf.precision(prediction, y, task='binary').cpu()
    recall = tmf.recall(prediction, y, task='binary').cpu()

    return running_loss / len(dataloader), acc, f1, prec, recall

def train(train_dataloader, test_dataloader, model, optim, config):
    
    model_name = config['model_name']
    dst = config['dst']
    if not os.path.exists(os.path.join(dst, model_name)):
        os.mkdir(os.path.join(dst, model_name))
    else:
        raise OSError('MODEL ALREADY EXISTS')

    # model = SimpleCNN()
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # model.fc = nn.Linear(512, 2)
    # model = transformers.ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    # model = transformers.Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    # model.classifier = nn.Linear(768, 2)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        num_devices = torch.cuda.device_count()
        devices = list(range(num_devices))

        print("Using devices", devices)
    print(device)
    if device != 'cpu':
        model = nn.DataParallel(model, device_ids=devices)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.MSELoss()

    best_f1 = 0
    best_acc = 0
    epochs = config['epochs']
    # loop
    valid_metrics = {
        'acc': [],
        'f1': [],
        'prec': [],
        'recall': []
    }
    best_metrics = {
        'acc': 0,
        'f1': 0,
        'prec': 0,
        'recall': 0
    }
    train_losses = []
    valid_losses = []
    optim_lrs = []
    for epoch in range(epochs):
        start = time.time()
        train_loss, lrs = train_step(model, train_dataloader, optim, loss_fn, device, show_progress=config['show_progress'], schedulers=config['schedulers'], accum_iter=config['accum_iter'])
        valid_loss, valid_acc, valid_f1, valid_prec, valid_recall = test_step(model, test_dataloader, loss_fn, device, show_progress=config['show_progress'])
        end = time.time()


        print(f'Epoch {epoch}/{epochs}:\nAvg Train Loss: {train_loss}\nAvg Valid Loss: {valid_loss}, Acc: {valid_acc}, F1 : {valid_f1}\nEpoch Time: {end - start}s\n')
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_metrics['acc'].append(valid_acc)
        valid_metrics['f1'].append(valid_f1)
        valid_metrics['prec'].append(valid_prec)
        valid_metrics['recall'].append(valid_recall)
        optim_lrs += lrs

        if valid_acc > best_metrics['acc']:
            torch.save(model.state_dict(), f'{dst}/{model_name}/best_model.pt')
            best_metrics = {
                'acc': valid_acc,
                'f1': valid_f1,
                'prec': valid_prec,
                'recall': valid_recall
            }

        plt.figure()
        plt.plot(train_losses, label='Train loss', color='green')
        plt.plot(valid_losses, label='Val loss', color='red')    
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.title("Loss over time")
        plt.legend(loc='upper center')
        plt.savefig(os.path.join(dst, model_name, f'loss.png'))
        plt.close()

        plt.figure()
        plt.plot(valid_metrics['acc'], label='Accuracy', color='green')
        plt.plot(valid_metrics['f1'], label='F1 score', color='red')    
        plt.xlabel("epochs")
        plt.ylabel("Score")
        plt.title("Validation accuracy and F1 over time")
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(dst, model_name, f'accf1.png'))
        plt.close()

        plt.figure()
        plt.plot(optim_lrs, label='LR', color='orange')
        plt.xlabel("iterations")
        plt.ylabel("Learning rate")
        plt.title("Scheduled learning rate")
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(dst, model_name, f'lr.png'))
        plt.close()

    return best_metrics
