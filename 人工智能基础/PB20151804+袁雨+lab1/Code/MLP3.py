# -*- coding = utf-8 -*-
# @Time:2022/4/29:54
# @Author:袁雨
# @File:MLP3.py
# @Software:PyCharm
import torch
from torch.utils.data import Dataset    # 抽象类 继承
from torch.utils.data import DataLoader     # 可实例化
import numpy as np
import pandas as pd
import torch.nn.functional as f   # 激励函数的库
import matplotlib.pyplot as plt
from sklearn import metrics


# 加载数据集
class GetData(Dataset):

    def __init__(self, mode):
        super(GetData, self).__init__()
        self.mode = mode  # 设置读取读取数据集的模式
        if mode == 'train':
            self.root = r'D:\桌面文件\lab3\insects-training.txt'  # 数据集存放的路径
        elif mode == 'test':
            self.root = r'D:\桌面文件\lab3\insects-testing.txt'
        else:
            print("error")
            self.root = r'insects-training.txt'
        data = pd.read_table(self.root)   # 从CSV文件中加载原始数据集
        self.label = data.iloc[:, -1]  # 装载标签
        self.data = np.array(data.iloc[:, 1:-1])    # 装载数据

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]


train_dataset = GetData('train')
print(train_dataset)
test_dataset = GetData('test')
# 创建加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True, num_workers=0)   # 并行进程数
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False, num_workers=0)

in_num = 285
out_num = 2


# 建立一个三层感知机网络
class MLP3(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self):
        super(MLP3, self).__init__()  #
        # 初始化三层神经网络 两个全连接的隐藏层，一个输出层
        hidden1_num = 256  # 第二层节点数
        hidden2_num = 256  # 第三层节点数
        self.fc1 = torch.nn.Linear(in_num, hidden1_num)  # 第一个隐含层
        self.fc2 = torch.nn.Linear(hidden1_num, hidden2_num)  # 第二个隐含层
        self.fc3 = torch.nn.Linear(hidden2_num, out_num)  # 输出层
        # 使用dropout防止过拟合
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, din):
        # 前向传播， 输入值：din, 返回值 dout
        # din = din.view(-1, 28 * 28)  # 将一个多行的Tensor,拼接成一行
        dout = f.relu(self.fc1(din.to(torch.float32)))  # 使用 relu 激活函数
        dout = self.dropout(dout)
        dout = f.relu(self.fc2(dout))
        dout = self.dropout(dout)
        dout = f.softmax(self.fc3(dout), dim=1)  # 输出层使用 softmax 激活函数
        # 10个数字实际上是10个类别，输出是概率分布，最后选取概率最大的作为预测值输出
        return dout


# 训练神经网络
# 循环次数
num_epoch = 20
# 一个batch的数据量
batch_size = 50
# 学习率
learning_rate = 0.1
# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train():
    # 定义损失函数
    lossfunc = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    # 定义优化器
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
    # 开始训练
    for epoch in range(num_epoch):
        train_loss = 0.0
        for i, (data, target) in enumerate(train_loader):   # enumerate得到循环次数
            print(data,target)
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()   # 清空上一步的残余更新参数值
            output = model(data)    # 得到预测值
            target = torch.as_tensor(target, dtype=torch.long)
            loss = lossfunc(output, target)  # 计算两者的误差
            loss.backward()         # 误差反向传播, 计算参数更新值
            optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
            train_loss += loss.item()*data.size(0)
        train_loss = train_loss / len(train_dataset)
        print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
        # 每遍历一遍数据集，测试一下准确率
        print('Accuracy of the network  on the train_data: %.6f %%' % test(train_loader))


# 在数据集上测试神经网络
def test(dataloader):
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集中不需要反向传播
        for item in dataloader:
            data, target = item
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    # print(100.0 * correct / total)
    return 100.0 * correct / total


# 声明感知机网络
model = MLP3()
model = model.cuda()


def main():
    # 声明感知机网络
    print("三层感知机：")
    train()
    print("验证集：")
    valid_pre = []
    valid_label = []
    for i, (data, label) in enumerate(test_loader):
        data = data.to(device)
        label = label.to(device)
        pre = model(data)  # 表示模型的预测输出
        test_prob.extend(pre[:, 1].cpu().detach().numpy())
        pre = pre.cpu().detach().numpy()  # 先把prob转到CPU上，然后再转成numpy，如果本身在CPU上训练的话就不用先转成CPU了
        test_pre.extend(np.argmax(pre, axis=1))  # 求每一行的最大值索引
        test_label.extend(label.cpu())
    print('Accuracy of the network  on the test_data: %.6f %%' % test(test_loader))
    print("F1 Score:{:.6f}".format(metrics.f1_score(test_label, test_pre)))
    fpr, tpr, thresholds = metrics.roc_curve(test_label, test_prob)
    auc = metrics.roc_auc_score(test_label, test_prob)
    print("AUC:{:.6f}".format(auc))
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(auc), lw=2)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  # 画对角线
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    # plt.savefig('your_path/model1.png', bbox_inches='tight') #这个是将生成的ROC曲线以图片保存下来，要把下面一行注释掉，这个就不会展示图片等你按q关掉，它会执行下去
    plt.show()

if __name__ == '__main__':
    main()



