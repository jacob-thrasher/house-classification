import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from data import get_train_test
import torchmetrics.functional as tmf

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
    loss = 0

    X, y = next(iter(dataloader))
    X = X.to(device)
    y = y.type(torch.float32).to(device)

    out = model(X)
    loss = loss_fn(out.squeeze(), y)
    loss = loss.item()

    acc = tmf.classification.accuracy(out.squeeze(), y, task='binary')
    f1 = tmf.f1_score(out.squeeze(), y, task='binary')


    return loss, acc, f1

def train(model_name='CNN'):
    print(torch.cuda.is_available())
    train_set, test_set = get_train_test(root='D:\\Big_Data\\zillow_images', csvpath='labels.csv')

    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=len(test_set), shuffle=True) 

    model = SimpleCNN()

    epochs = 150
    device = 'cpu'
    if torch.cuda.is_available():
        print("Sending to cuda device")
        device='cuda'
    model.to(device)

    # optim = Adam(model.parameters(), lr=0.001)
    optim = SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.BCELoss()

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

        print(f'Epoch {epoch}/{epochs}:\nAvg Train Loss: {train_loss}\nAvg Valid Loss: {valid_loss}, Acc: {valid_acc}, F1 : {valid_f1}\nEpoch Time: {end - start}s\n')
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc.cpu())
        valid_f1s.append(valid_f1.cpu())

        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), f'{model_name}.pt')

    plt.figure()
    plt.plot(train_losses, label='Train loss', color='green')
    plt.plot(valid_losses, label='Val loss', color='red')    
    plt.xlabel("epochs")
    plt.ylabel("BCE Loss")
    plt.title("Loss over time")
    plt.legend(loc='upper center')
    plt.savefig('loss.png')

    plt.figure()
    plt.plot(valid_accs, label='Accuracy', color='green')
    plt.plot(valid_f1s, label='F1 score', color='red')    
    plt.xlabel("epochs")
    plt.ylabel("Score")
    plt.title("Validation accuracy and F1 over time")
    plt.legend(loc='lower right')
    plt.savefig('acc_f1.png')

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