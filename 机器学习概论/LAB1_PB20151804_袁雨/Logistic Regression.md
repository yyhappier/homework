# Logistic Regression

*袁雨* *PB20151804*



## 一、实验内容

In `Logistic.py`, write your own Logistic Regression class

In `Load.ipynb`

1. Deal with NULL rows, you can either choose to drop them or replace them with mean or other value
2. Encode categorical features
3. Split the dataset into X_train, X_test, y_train, y_test, also you can then use normalization or any other methods you want
4. Train your model and plot the loss curve of training
5. Compare the accuracy(or other metrics you want) of test data with different parameters you train with, i.e. learning rate, regularization methods and parameters .etc



## 二、实验要求

- Do not use sklearn or other machine learning library, you are only permitted with numpy, pandas, matplotlib, and [Standard Library](https://docs.python.org/3/library/index.html), you are required to write this project from scratch.
- You are allowed to discuss with other students, but you are not allowed to plagiarize the code, we will use automatic system to determine the similarity of your programs, once detected, both of you will get zero mark for this project.
- Report
  - The Loss curve of one training process
  - The comparation table of different parameters
  - The best accuracy of test data



## 三、实验步骤

1. #### 数据集分析

   ```python
   df.head()
   ```

   ![image-20221015225834666](D:\桌面文件\image-20221015225834666.png)

   ​		查看数据集的前五行。

   

   ```python
   df.info()
   ```

   <img src="C:\Users\袁雨\AppData\Roaming\Typora\typora-user-images\image-20221013203919172.png" alt="image-20221013203919172" style="zoom:50%;" />

   ​		查看数据集基本信息。该数据集一共13列，包括12个特征和1个目标。一共614行。特征有int、float、object等类型，包含缺失值。

   

   ```python
   df['Loan_Status'].value_counts()
   ```

   <img src="C:\Users\袁雨\AppData\Roaming\Typora\typora-user-images\image-20221013203947550.png" alt="image-20221013203947550" style="zoom:50%;" />

   ​		观察目标列，包含正例422个，负例192个。

   

2. #### 数据预处理

   （1）特征选择

   ​		去掉在预测目标上没有意义的特征"Loan_ID"列，再统计缺失值。

   ```python
   df.drop("Loan_ID", axis=1, inplace=True)
   # Checking the Missing Values
   df.isnull().sum()
   ```

   <img src="C:\Users\袁雨\AppData\Roaming\Typora\typora-user-images\image-20221013204109144.png" alt="image-20221013204109144" style="zoom: 45%;" />

   

   （2）特征编码

   ​		将object型的分类特征转换成数字型，方便机器学习模型计算。

   ```python
   GenderMap = {'Male':1,'Female':0}
   MarriedMap = {'Yes':1,'No':0}
   DependentsMap={'3+':3,'2':2,'1':1,'0':0}
   EducationMap = {'Graduate':1,'Not Graduate':0}
   Self_EmployedMap = {'Yes':1,'No':0}
   Property_AreaMap = {'Urban':2,'Semiurban':1,'Rural':0}
   Loan_StatusMap = {'Y':1,'N':0}
   df['Gender'] = df['Gender'].map(GenderMap)
   ……
   ```

   ![image-20221015173950675](D:\桌面文件\image-20221015173950675.png)

   

   （3）缺失值处理

   ​		可以丢弃缺失值、用均值填充等。实验证明若丢弃缺失值则只剩下480个样本，训练效果较差，故选择均值填充。

   ```python
   # Task1 deal with NULL rows, you can either choose to drop them or replace them with mean or other value
   for column in list(df.columns[df.isnull().sum() > 0]):
       df[column].fillna(df[column].mean(), inplace=True)
   ```

   ![image-20221015211922643](D:\桌面文件\image-20221015211922643.png)

   ​		

   ​		数据清理后查看数据集的若干统计值。

   ```python
   df.describe()
   ```

   ![image-20221015173527421](D:\桌面文件\image-20221015173527421.png)

   ​		查看正例和负例不同特征的均值差异。

   ```python
   df.groupby('Loan_Status').mean()
   ```

   ![image-20221015173542491](D:\桌面文件\image-20221015173542491.png)

   

   （4）标准化

   ​		可以使用z-score、min_max等标准化方法。因为Logistic Regression的参数优化采用了梯度下降法，如果不对特征进行归一化，可能会使得损失函数值得等高线呈椭球形，这样花费更多的迭代步数才能到达最优解；损失函数加入了正则项，参数的大小决定了损失函数值等。经实验验证，min_max归一化方法效果更好。故对特征进行归一化。

   ```python
   X=df.iloc[:,:-1]
   X_MinMax = (X-X.min())/(X.max()-X.min())
   print(X_MinMax)
   ```

   <img src="D:\桌面文件\image-20221015174042207.png" alt="image-20221015174042207" style="zoom: 50%;" />

   ​		

3. #### 实现Logistic Regression

   （1）模型

   ​		$\hat y=\frac{1}{1+e^{-z}}$，$z=wX+b$

   ​		$\hat y$：预测值。使用sigmoid 函数，表示$P（y=1|X)$，y=1的概率；

   ​		$X$：输入特征；

   ​		$w$：weight，权重；

   ​		$b$：bias，偏差。

   

   （2）优化目标

   ​		使用交叉熵损失函数，并加入正则化。

   ​		L1正则化：$J(w,b) =  -\frac{1}{m}\sum_{i=1}^{m}[y_i\log(\hat y_i)+(1-y_i)log(1-\hat y_i)]+\frac{1}{m}\gamma||w||_{1}$

   ​		L2正则化： $ J(w,b)=-\frac{1}{m}\sum_{i=1}^{m}[y_i\log(\hat y_i)+(1-y_i)log(1-\hat y_i)]+\frac{1}{m}\gamma w^Tw$

   ​		$m$：样本数。

   ​		$\gamma$：正则化参数。

   

   （3）优化方法

   ​		使用梯度下降法更新参数$w$和$b$。

   ​		$w=w-lr\frac{\partial L}{\partial w}$

   ​		$b=b-lr\frac{\partial L}{\partial b}$

   ​		$\frac{\partial J}{\partial w}=-\frac{1}{m}X^T(\hat y-y)+\begin{cases}\frac{1}{m}\gamma \sum_{i=1}^m sign(w_i)\quad \text {if L1}  \\
   \frac{1}{m}\gamma w\quad \text{if L2}\\
   \end{cases}$

   ​		$\frac{\partial J}{\partial b} = -\frac{1}{m}(\hat y-y)$

   ​		$lr$：learning rate，学习率。

   

   （4）算法实现

   ​		见`Logistic.py`。



3. #### 训练

   ​		使用`Logistic.py`里实验的LogisticRegression进行训练，返回训练得到的分类器、迭代次数与cost列表。

   ​		使用迭代次数与cost列表作出loss curve。

   

4. #### 测试

   ​		使用训练得到的分类器进行测试，采用十折交叉验证，最终得到十轮结果的accuracy均值。分别保存训练集与测试集的accuracy，以比较判断是否过拟合等。

   

## 四、实验结果

1. #### The Loss curve of one training process

   （penalty="l2"，gamma=6，fit_intercept=True，lr=0.01，tol=1e-7，max_iter=1e4）

   <img src="D:\桌面文件\image-20221015223840109.png" alt="image-20221015223840109" style="zoom:80%;" />

   

2. #### The comparation table of different parameters

   初始：penalty="l2"，gamma=10，fit_intercept=True，lr=0.01，tol=1e-7，max_iter=1e4

   （1）预处理方法

   ① 标准化方法

   | 标准化方法        | training accuracy  | testing accuracy   |
   | ----------------- | ------------------ | ------------------ |
   | 归一化（min-max） | 0.8081967213114754 | 0.8095840867992766 |
   | 正规化（z-score） | 0.7901639344262296 | 0.7867992766726943 |

   ​		可见归一化得到的accuracy更高，故选择归一化。

   

   ② 缺失值处理

   | 缺失值处理 | training accuracy  | testing accuracy   |
   | ---------- | ------------------ | ------------------ |
   | 丢弃       | 0.8083333333333333 | 0.8083333333333332 |
   | 均值填充   | 0.8081967213114754 | 0.8095840867992766 |

   ​		丢弃之后只剩下480个样本。均值填充的样本数更多，得到的accuracy更高，故选择均值填充。

   

   （2）调参

   ① penalty

   ​		l1：L1正则化；
   ​		l2：L2正则化。

   | penalty | training accuracy  | testing accuracy   |
   | ------- | ------------------ | ------------------ |
   | l1      | 0.8081967213114754 | 0.8095840867992766 |
   | l2      | 0.8081967213114754 | 0.8095840867992766 |

   ​		可见二者最终的accuracy相同。

   ​		观察二者的loss curve：

   ​		l1：

   ![download (1)](D:\桌面文件\download (1).png)

   ​		l2：

   ![download](D:\桌面文件\download.png)

   ​		L2正则化的loss curve下降略快，且更平滑，故选择l2。

   

   ② gamma

   ​		正则化参数。

   | gamma | training accuracy  | testing accuracy   |
   | ----- | ------------------ | ------------------ |
   | 0     | 0.8065573770491803 | 0.8092224231464737 |
   | 3     | 0.8081967213114754 | 0.8095840867992766 |
   | 6     | 0.8081967213114754 | 0.8095840867992766 |
   | 10    | 0.8081967213114754 | 0.8095840867992766 |
   | 20    | 0.8081967213114754 | 0.8095840867992766 |
   | 30    | 0.7606557377049181 | 0.7712477396021701 |

   ​		正则化参数取3~20左右时accuracy达最高，选择 gamma = 6。

   

   ③ fit_intercept

   ​			True：包含bias；

   ​			False：不包含bias。

   | fit_intercept | training accuracy  | testing accuracy   |
   | ------------- | ------------------ | ------------------ |
   | True          | 0.8081967213114754 | 0.8095840867992766 |
   | False         | 0.8081967213114754 | 0.8095840867992766 |

   ​		当max_iter = 10000, tol = 1e-7时，二者的accuracy相同。

   ​		接着比较当max_iter = 3000时：

   | fit_intercept | training accuracy  | testing accuracy   |
   | ------------- | ------------------ | ------------------ |
   | True          | 0.8081967213114754 | 0.8090415913200723 |
   | False         | 0.8032786885245902 | 0.8057866184448462 |

   ​		可见fit_intercept = True时的accuracy略高，取fit_intercept = True。

   

   ④ lr

   ​		学习率。

   | lr    | training accuracy  | testing accuracy   |
   | ----- | ------------------ | ------------------ |
   | 0.001 | 0.6868852459016394 | 0.6873417721518987 |
   | 0.01  | 0.8081967213114754 | 0.8095840867992766 |
   | 0.1   | 0.8081967213114754 | 0.8095840867992766 |

   ​		可见学习率过低会导致收敛速度变慢，取lr=0.01。

   

   ⑤ tol

   ​		停止迭代的tolerance。

   | tol       | training accuracy  | testing accuracy   |
   | --------- | ------------------ | ------------------ |
   | $10^{-5}$ | 0.8081967213114754 | 0.8079566003616637 |
   | $10^{-7}$ | 0.8081967213114754 | 0.8095840867992766 |
   | $10^{-9}$ | 0.8081967213114754 | 0.8095840867992766 |

   ​		tol < $10^{-7}$时，accuracy几乎不变化，故没必要继续减小。取 tol = 1e-7。

   

   ⑥ max_iter

   ​		最大迭代次数。

   | max_iter | training accuracy  | testing accuracy   |
   | -------- | ------------------ | ------------------ |
   | 1000     | 0.6916666666666667 | 0.6916666666666667 |
   | 3000     | 0.8081967213114754 | 0.8090415913200723 |
   | 5000     | 0.8081967213114754 | 0.8095840867992766 |
   | 10000    | 0.8081967213114754 | 0.8095840867992766 |
   | 20000    | 0.8081967213114754 | 0.8095840867992766 |

   ​		max_iter = 10000后，accuracy几乎不变化，故没必要继续增大。取max_iter = 10000。

   ​	

   ​		总得来说该模型收敛较快，调参起的作用不大。

   

3. #### The best accuracy of test data

   0.8095840867992766

   （penalty="l2"，gamma=6，fit_intercept=True，lr=0.01，tol=1e-7，max_iter=1e4）