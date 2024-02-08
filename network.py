import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from data import get_train_test
import torchmetrics.functional as tmf
import os
from torchvision import models
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torch.optim.lr_scheduler import CosineAnnealingLR

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

def train_step(model, dataloader, optim, loss_fn, device):
    model.train()
    epoch_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.type(torch.float32).to(device)

        optim.zero_grad()
        out = model(X)
        # loss = loss_fn(out.logits.squeeze(), y)
        loss = loss_fn(out.squeeze(), y)
        epoch_loss += loss.item()
        loss.backward()
        optim.step()

    return epoch_loss / len(dataloader)

def test_step(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0

    for (X, y) in (dataloader):
        X = X.to(device)
        y = y.type(torch.float32).to(device)

        out = model(X)
        loss = loss_fn(out.squeeze(), y)
        running_loss += loss.item()

    # acc = tmf.classification.accuracy(out.squeeze(), y, task='binary')
    # f1 = tmf.f1_score(out.squeeze(), y, task='binary')

    acc = 0
    f1 = 0
    return running_loss / len(dataloader), acc, f1

def train(train_dataset, test_dataset, model_name='CNN'):
    print(torch.cuda.is_available())
    # train_set, test_set = get_train_test(root='D:\\Big_Data\\zillow_images', csvpath='labels.csv')
    

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False) 

    # model = SimpleCNN()
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.fc = nn.Linear(512, 1)

    epochs = 150
    device = 'cpu'
    if torch.cuda.is_available():
        print("Sending to cuda device")
        device='cuda'
    model.to(device)

    optim = Adam(model.parameters(), lr=0.1, weight_decay=0.01)
    # scheduler = CosineAnnealingLR(optim, T_max=150)

    # loss_fn = nn.BCELoss()
    loss_fn = nn.MSELoss()

    best_valid_loss = 1e10
    # # loop
    train_losses = []
    valid_losses = []
    valid_accs = []
    valid_f1s = []
    for epoch in range(epochs):
        start = time.time()
        train_loss = train_step(model, train_dataloader, optim, loss_fn, device)
        valid_loss, valid_acc, valid_f1 = test_step(model, test_dataloader, loss_fn, device)
        end = time.time()

        # scheduler.step()

        print(f'Epoch {epoch}/{epochs}:\nAvg Train Loss: {train_loss}\nAvg Valid Loss: {valid_loss}, Acc: {valid_acc}, F1 : {valid_f1}\nEpoch Time: {end - start}s\n')
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        valid_f1s.append(valid_f1)

        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), f'models/{model_name}.pt')

        plt.figure()
        plt.plot(train_losses, label='Train loss', color='green')
        plt.plot(valid_losses, label='Val loss', color='red')    
        plt.xlabel("epochs")
        plt.ylabel("MSE Loss")
        plt.title("Loss over time")
        plt.legend(loc='upper center')
        plt.savefig(os.path.join('metrics', model_name, 'loss.png'))
        plt.close()

        # plt.figure()
        # plt.plot(valid_accs, label='Accuracy', color='green')
        # plt.plot(valid_f1s, label='F1 score', color='red')    
        # plt.xlabel("epochs")
        # plt.ylabel("Score")
        # plt.title("Validation accuracy and F1 over time")
        # plt.legend(loc='lower right')
        # plt.savefig(os.path.join('metrics', model_name, 'acc_f1.png'))

# print("CUDA:", torch.cuda.is_available())
# torch.manual_seed(1)
# train()

# model = CNN()
# model.load_state_dict(torch.load('CNN.pt'))
# model.to('cuda')
# model.eval()


# train_set, test_set = get_train_test(root='D:\\Big_Data\\zillow_images', csvpath='labels.csv')
# test_dataloader = DataLoader(test_set, batch_size=len(test_set), shuffle=True) 

# loss_fn = nn.BCELoss()
# valid_loss, valid_acc, valid_f1 = test_step(model, test_dataloader, loss_fn, 'cuda')
# print(f'Avg Valid Loss: {valid_loss}, Acc: {valid_acc}, F1 : {valid_f1}')