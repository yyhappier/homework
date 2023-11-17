# -*- coding = utf-8 -*-
# @Time:2023/3/2121:43
# @Author:袁雨
# @File:main.py.py
# @Software:PyCharm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 自定义生成数据集
class CustomDataset(Dataset):
    def __init__(self, low, high, size, device, dtype):
        self.X = torch.linspace(low, high, size, device=device, dtype=dtype).unsqueeze(1)
        self.Y = torch.sin(self.X) + torch.cos(self.X) + torch.sin(self.X) * torch.cos(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].to(device), self.Y[idx].to(device)


# 前馈神经网络
class FeedforwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_func):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation_func,
            nn.Linear(hidden_size, hidden_size),
            activation_func,
            nn.Linear(hidden_size, hidden_size),
            activation_func,
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


# 训练
def train(net, data_loader, val_X, val_Y, lr, epochs, patience):
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=len(data_loader), gamma=0.5)

    train_loss_history = []
    val_loss_history = []

    best_val_loss = float('inf')
    relative_train_loss = float('inf')
    best_model_params = net.state_dict()
    early_stop = 0
    for epoch in range(epochs):
        train_loss_epoch = 0
        for X_batch, Y_batch in data_loader:
            optimizer.zero_grad()
            output = net(X_batch)
            train_loss = f.mse_loss(output, Y_batch)
            train_loss.backward()
            optimizer.step()
            train_loss_epoch += train_loss.item()
        with torch.no_grad():
            output_val = net(val_X)
            val_loss = f.mse_loss(output_val, val_Y)

        train_loss_history.append(train_loss_epoch / len(data_loader))
        val_loss_history.append(val_loss.item())

        if epoch % 10 == 9:
            print(
                f"Epoch: {epoch + 1}, Train Loss: {train_loss_epoch / len(data_loader):.8f}, Val Loss: {val_loss.item():.8f}")

        scheduler.step()

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            relative_train_loss = train_loss_epoch / len(data_loader)
            best_model_params = net.state_dict()
            early_stop = 0
        else:
            early_stop += 1
            if early_stop == patience:
                print(f"Early stopping reached at epoch {epoch}.")
                print(f"best_val_loss:{best_val_loss:.8f}")
                print(f"relative_train_loss:{relative_train_loss:.8f}")
                break

    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss")
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("loss.jpg")
    plt.show()

    net.load_state_dict(best_model_params, strict=False)

    return net


# 测试
def test(net, test_X, test_Y):
    net.to(device)

    with torch.no_grad():
        output_test = net(test_X)
        test_loss = f.mse_loss(output_test, test_Y)

    print(f"Test Loss: {test_loss.item():.8f}")

    plt.scatter(test_X.cpu(), test_Y.cpu())
    plt.scatter(test_X.cpu(), output_test.cpu().detach())
    plt.title("test")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(['Actual', 'Predicted'])
    plt.savefig("test.jpg")
    plt.show()


# 参数
input_size = 1
hidden_size = 16
output_size = 1
activation_func = nn.Softplus()

lr = 0.01
epochs = 3000
batch_size = 64
dtype = torch.float
train_size = 6000
valid_size = 2000
test_size = 2000
patience = 100

# 生成数据集
dataset = CustomDataset(0, 2 * np.pi, train_size + valid_size + test_size, device, dtype)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_X, val_Y = val_dataset[:]
test_X, test_Y = test_dataset[:]

# 定义网络
net = FeedforwardNet(input_size, hidden_size, output_size, activation_func)

# 训练
net = train(net, train_loader, val_X, val_Y, lr, epochs, patience)
# 测试
test(net, test_X, test_Y)
