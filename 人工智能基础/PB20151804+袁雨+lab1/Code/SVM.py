# -*- coding = utf-8 -*-
# @Time:2022/4/312:56
# @Author:袁雨
# @File:SVM.py
# @Software:PyCharm
from sklearn import svm
import numpy as np
from sklearn.metrics import f1_score
import gc
import pandas as pd
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
# from thundersvm import SVC

class GetData():

    def __init__(self, mode):
        super(GetData, self).__init__()
        self.mode = mode  # 设置读取读取数据集的模式
        if mode == 'train':
            self.root = r'D:\PytorchProject\lab1\Lab1_train.csv'  # 数据集存放的路径
        elif mode == 'val':
            self.root = r'D:\PytorchProject\lab1\Lab1_validation.csv'
        elif mode == 'test':
            self.root = r'D:\PytorchProject\lab1\Lab1_test.csv'
        else:
            print("error")
            self.root = r'D:\PytorchProject\lab1\Lab1_train.csv'
        data = pd.read_csv(self.root)
        self.label = np.array(data.iloc[:, -1]).tolist()  # 将数据转为list格式
        self.data = np.array(data.iloc[:, 1:-1]).tolist()

    def getitem(self):
        return self.data, self.label


def main():
    print("训练集：")
    train_data, train_label = GetData('train').getitem()
    train_data = train_data[0:50]     # 抽取10000行小样本训练
    train_label = train_label[0:50]
    classifier = svm.SVC(C=0.75, kernel='rbf', decision_function_shape='ovr')  # ovr:一对多策略  linear rbf
    clf_proba = classifier.fit(train_data, train_label)
    print("准确率：%0.6f" % classifier.score(train_data, train_label))
    del train_data, train_label
    gc.collect()
    print("验证集：")
    val_data, val_label = GetData('val').getitem()
    print("准确率：%0.6f" % classifier.score(val_data, val_label))
    print("F1 score = {:.6f}".format(f1_score(val_label, classifier.predict(val_data))))
    del val_data, val_label
    gc.collect()
    print("测试集：")
    test_data, test_label = GetData('test').getitem()
    # 4.计算svm分类器的准确率
    print("准确率：%0.6f" % classifier.score(test_data, test_label))
    print("F1 score = {:.6f}".format(f1_score(test_label, classifier.predict(test_data))))
    # print(clf_proba.decision_function(test_data))
    fpr, tpr, thresholds = roc_curve(test_label, clf_proba.decision_function(test_data))
    AUC = auc(test_label, clf_proba.decision_function(test_data))
    print("AUC:%0.6f" % AUC)
    plt.figure()
    plt.plot(fpr, tpr, 'k--', label='ROC (area = %0.2f)' % AUC, lw=2)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  # 画对角线
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    main()
