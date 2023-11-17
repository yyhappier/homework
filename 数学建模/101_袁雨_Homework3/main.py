# -*- coding = utf-8 -*-
# @Author:袁雨
# @File:main.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 加载数据
train_data = np.loadtxt(r"insects-training.txt")
train_x, train_y = train_data[:, :2], train_data[:, 2].astype(int)

# 划分训练集与验证集
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
train_x, valid_x = torch.from_numpy(train_x).float(), torch.from_numpy(valid_x).float()
train_y, valid_y = torch.from_numpy(train_y).long(), torch.from_numpy(valid_y).long()

test_data = np.loadtxt(r"insects-testing.txt")
test_x, test_y = test_data[:, :2], test_data[:, 2].astype(int)

test_x_1, test_y_1 = test_x[:60], test_y[:60]
test_x_1 = torch.from_numpy(test_x_1).float()
test_y_1 = torch.from_numpy(test_y_1).long()

test_x_2, test_y_2 = test_x[60:], test_y[60:]
test_x_2 = torch.from_numpy(test_x_2).float()
test_y_2 = torch.from_numpy(test_y_2).long()


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 3)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)

        return x


# 训练
def train(model, optimizer, scheduler, criterion, train_x, train_y, valid_x, valid_y, max_iter, early_stop_patience):
    train_loss_list = []
    valid_loss_list = []
    best_valid_loss = float('inf')  # 初始化最优验证集loss
    early_stop_counter = 0  # 初始化早停计数器

    for epoch in range(max_iter):
        model.train()  # 模型训练模式
        optimizer.zero_grad()
        outputs = model(train_x)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
        scheduler.step()  # 更新学习率

        # 计算训练集accuracy和loss
        train_acc = acc(outputs, train_y)
        train_loss_list.append(loss.item())

        with torch.no_grad():
            model.eval()  # 模型评估模式
            valid_outputs = model(valid_x)
            valid_loss = criterion(valid_outputs, valid_y)
            valid_acc = acc(valid_outputs, valid_y)
            valid_loss_list.append(valid_loss.item())

            # 如果当前验证集loss更优，则更新最优验证集loss并重置早停计数器，否则增加早停计数器
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_train_loss = loss
                early_stop_counter = 0
                best_train_acc = train_acc
                best_valid_acc = valid_acc
                best_train_outputs = outputs
                best_valid_outputs = valid_outputs
                best_model_state_dict = model.state_dict()
            else:
                early_stop_counter += 1

            # 如果连续early_stop_patience个epoch验证集loss没有改善，则停止训练
            if early_stop_counter == early_stop_patience:
                print('early stopping')
                break

        if (epoch + 1) % 10 == 0:
            print(
                'epoch [{}/{}], training loss: {:.6f}, validation loss: {:.6f}'.format(epoch + 1, max_iter, loss.item(),
                                                                                       valid_loss.item()))

    print('best training loss: {:.6f}, best validation loss: {:.6f}'.format(best_train_loss.item(),
                                                                            best_valid_loss.item()))
    print('best training acc:{:.2f} %, best validation acc:{:.2f} %'.format(best_train_acc, best_valid_acc))
    return best_model_state_dict, train_loss_list, valid_loss_list, torch.max(best_train_outputs, 1)[1].numpy(), \
           torch.max(best_valid_outputs, 1)[1].numpy()


# 测试
def test(model, test_x, test_y):
    test_inputs = test_x
    test_outputs = model(test_inputs)
    test_acc = acc(test_outputs, test_y)

    return test_acc, torch.max(test_outputs, 1)[1].numpy()


def acc(outputs, y):
    _, predicted = torch.max(outputs.data, dim=1)
    total = len(y)
    correct = (predicted == y).sum().item()
    return 100 * correct / total


def plot_data(x, y, figname):
    plt.scatter(x[y == 0, 0], x[y == 0, 1], c='coral', label='class 0')
    plt.scatter(x[y == 1, 0], x[y == 1, 1], c='seagreen', label='class 1')
    plt.scatter(x[y == 2, 0], x[y == 2, 1], c='skyblue', label='class 2')
    plt.title(figname)
    plt.xlabel('Body length')
    plt.ylabel('Wing length')
    plt.legend(loc='lower right')
    plt.savefig(figname + '.png')
    plt.close()


def plot_loss(train_loss_list, valid_loss_list):
    plt.plot(train_loss_list, label='training')
    plt.plot(valid_loss_list, label='validation')
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss.png')


model = Net()

# 定义损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# 定义学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

best_model_state_dict, train_loss_list, valid_loss_list, train_pred, valid_pred = train(
    model, optimizer, scheduler, criterion, train_x, train_y, valid_x, valid_y, max_iter=5000, early_stop_patience=1500)

model.load_state_dict(best_model_state_dict)

test_acc_1, test_pred_1 = test(model, test_x_1, test_y_1)
test_acc_2, test_pred_2 = test(model, test_x_2, test_y_2)
print('testing accuracy on test1: {:.2f} %'.format(test_acc_1))
print('testing accuracy on test2: {:.2f} %'.format(test_acc_2))

# plot_data(train_x, train_y, 'train')
# plot_data(valid_x, valid_y, 'valid')
# plot_data(train_x, train_pred, 'train_pred')
# plot_data(valid_x, valid_pred, 'valid_pred')
plot_data(test_x_1, test_y_1, 'test1')
plot_data(test_x_2, test_y_2, 'test2')
plot_data(test_x_1, test_pred_1, 'test_pred1')
plot_data(test_x_2, test_pred_2, 'test_pred2')

plot_loss(train_loss_list, valid_loss_list)