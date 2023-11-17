# LAB4 Density Peak Clustering

*袁雨 PB20151804*



## 一、实验目的

​	本次实验需要实现《**Clustering by fast search and find of density peaks**》一文中的算法（以下简称**DPC**）。总体流程是完成 DPC 算法的代码实现，并在给定数据集上进行可视化实验。



## 二、实验原理

### 算法思想

集成了 k-means 和 DBSCAN 两种算法的思想

- 聚类中心周围密度较低，中心密度较高
- 聚类中心与其它密度更高的点之间通常都距离较远

### 算法流程

1. Hyperparameter: a distance threshold $d_c$
2. For each data point $i$, compute two quantities:
   - Local density: $\rho_i = \sum_{j}\chi(d_{ij}-d_c)$,where $\chi(x)=1$ if $x<0$ and $\chi(x)=0$ otherwise
   - Distance from points of higher density: $\delta_i=\mathop {min}\limits_{j:\rho_j>\rho_i} d_{ij}$   · For the point with highest density, take $\delta_i=\mathop {max}\limits_{j} d_{ij}$
3. Identify the cluster centers and out-of-distribution (OOD) points

- Cluster centers: with both high $\rho_i$ and $\delta_i$
- OOD points: with high $\delta_i$ but low $\rho_i$
- Draw a decision graph, and make decisions manually



## 三、实验数据

本次实验采用 3 个 2D 数据集（方便可视化）

- Datasets/D31.txt
- Datasets/R15.txt
- Datasets/Aggregation.txt

数据格式

- 每个文件都是普通的 txt 文件，包含一个数据集
- 每个文件中，每一行表示一条数据样例，以空格分隔

注意事项

- 允许对不同的数据集设置不同的超参数

  

## 四、实验任务及要求

### （一）任务

#### 实验简介

本次实验的总体流程是完成 DPC 算法的代码实现，并在给定数据集上进行可视化实验。具体来说，同学们需要实现以下步骤

1. 读取数据集，（如有必要）对数据进行预处理
2. 实现 DPC 算法，计算数据点的 $\delta_i$ 和 $\rho_i$
3. 画出$\color{red}{决策图}$，选择样本中心和异常点
4. 确定分簇结果，计算$\color{red}{评价指标}$，画出$\color{red}{可视化图}$

助教除大致浏览代码外，以以下输出为评价标准：

- 可视化的决策图
- 可视化的聚类结果图
- 计算出的评价指标值 （DBI）
- 输出只要在**合理范围**内即可，不作严格要求

实验结果需要算法代码和实验报告

- 助教将通过可视化结果和代码来确定算法实现的正确性
- 助教将阅读实验报告来检验同学对实验和算法的理解

#### 评价指标

- 本次实验采用 Davis-Bouldin Index (DBI) 作为评价指标
- 建议$\color{red}{统一调用}$ sklearn.metrics. davies_bouldin_score 进行计算

#### 数据可视化

- 本次实验需要画两个二维散点图：决策图和聚类结果图
- 可视化库推荐 pyplot (也可自行选择别的工具，此处只做教程)

### （二）要求

- 禁止使用` sklearn` 或者其他的机器学习库，你只被允许使用`numpy`, `pandas`, `matplotlib`, 和 [Standard Library](https://gitee.com/link?target=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Findex.html), 你需要从头开始编写这个项目。
- 你可以和其他同学讨论，但是你不可以剽窃代码，我们会用自动系统来确定你的程序的相似性，一旦被发现，你们两个都会得到这个项目的零分。



## 五、实验步骤

1. 读取数据集，对数据进行预处理

   ```python
   dfA = pd.read_csv('Datasets//Aggregation.txt', header=None , sep = ' ')
   dfA = np.array((dfA-dfA.min())/(dfA.max()-dfA.min()))
   ```

   ​		以数据集Aggregation.txt为例。使用`pd.read_csv`进行读取，因为源文件中没有列索引，故设置`header=None`。数据集中列以空格相隔，故设置分隔符`seq=' '`。

   ​		对数据进行预处理，将其进行Min_Max标准化，消除特征量纲等的影响，并将数据类型转换为numpy数组。

   

2. 实现 DPC 算法

   （1）计算距离$d_{ij}$

   ```python
   def GetDistance(self):
       self.d = np.zeros([self.m,self.m])
       for i in range(self.m):
           for j in range(i+1,self.m):
               self.d[i][j] = np.sqrt(np.sum((self.X[i] - self.X[j]) ** 2))
               self.d[j][i] = self.d[i][j]
   ```

   ​		使用欧氏距离，计算距离$d_{ij}(i<j)$，并令$d_{ji}=d_{ij}$，保存在二维数组`d`中。

   

   （2）确定截断距离$d_c$

   ```python
   dA = np.triu(dpc1.d,0)
   dA = dA[np.where(dA!=0)].flatten()
   dA = np.sort(dA)
   print(dA[int(len(dA)*0.01)])
   print(dA[int(len(dA)*0.02)])
   ```

   ​		论文中提供了对$d_c$选取的一个建议： As a rule of thumb, one can choose dc so that the average number of neighbors is around 1 to 2% of the total number of points in the data set.

   ​		故取`d`中`i<j`的部分，并对其进行升序排序，取前1%~2%的数作为$d_c$，实际调参时根据决策图与聚类结果等再做调整与选择。

   

   （3）计算局部密度$\rho_i$

   ```python
   def GetRho(self):
       self.rho = np.zeros(self.m)
       if self.kernel == 'cut-off':
           dd = self.d - self.dc
           for i,di in enumerate(dd):
               self.rho[i] = di[di<0].size
       if self.kernel == 'gauss':
           self.rho = np.sum(np.exp(-self.d/self.dc) ** 2, axis = 1)
   ```

   ​		论文中使用Cut-off kernel： $\rho_i = \sum_{j}\chi(d_{ij}-d_c)，\chi(x)= \begin{cases}1, & x<0 \\ 0, & x \geq 0\end{cases}$

   ​		论文中还提到了Gaussian kernel：$\rho_i=\sum_{j} e^{-\left(\frac{d_{i j}}{d_c}\right)^2}$。

   ​		其中 cut-off kernel 为离散值，Gaussian kernel 为连续值，故后者产生冲突（不同的数据点具有相同的局部密度值）的概率更小。且 Gaussian kernel 满足：与$x_i$距离小于$d_c$的数据点越多，$\rho_i$值越大。

   ​		将计算结果保存在数组`rho`中。

   

   （4）计算距离 $\delta_i$

   ```python
   def GetDelta(self):
       self.delta = np.zeros(self.m)
       sort_index = np.argsort(-self.rho)
       self.index = sort_index
       self.delta[sort_index[0]] = np.max(self.d[sort_index[0]])
       for i in range(1,self.m):
           self.delta[sort_index[i]] = np.min(self.d[sort_index[i]][sort_index[0:i]])
   ```

   ​		论文中对$\delta_i$的计算方法为：$\delta_i$ is measured by computing the minimum distance between the point $i$ and any other point with higher density: $\delta_i=\mathop {min}\limits_{j:\rho_j>\rho_i} d_{ij}$. For the point with highest density, take $\delta_i=\mathop {max}\limits_{j} d_{ij}$		

   ​		使用`sort_index`（以下用s代替）数组保存 $\left\{\rho_i\right\}_{i=1}^m$ 的一个降序排列下标序，则可定义
   $$
   \delta_{s_i}= \begin{cases}\min _{\substack{j<i}}\left\{d_{s_i s_j}\right\}, & i \geq 1 \\ \max _{j}\left\{d_{s_0,j}\right\}, & i=0 .\end{cases}
   $$
   ​		将计算结果保存在数组`delta`中。

   

   （5）确定聚类中心和异常点

   ```Python
   self.c = np.zeros(self.m)
       self.center = []
       cn = 1
       for i in range(self.m):
           # 聚类中心
           if self.rho[i]>=self.thr_rho and self.delta[i]>=self.thr_delta:
               self.center.append(i)
               self.c[i]=cn
               cn+=1
           # 异常点
           elif self.rho[i]<self.thr_rho and self.delta[i]>=self.thr_delta:
               self.c[i] = -1
   
   print(f'共{cn}个cluster')
   ```

   - Cluster centers: with both high $\rho_i$ and $\delta_i$
   - OOD points: with high $\delta_i$ but low $\rho_i$

   ​		将$\rho>=thr\_rho，\delta>=thr\_delta$ 的点定义为聚类中心，将 $\rho<thr\_rho，\delta>=thr\_delta$ 的点定义为异常点，其中$thr\_rho，thr\_delta$ 根据决策图人工选择。

   ​		使用数组`c`保存数据点归类属性标记，定义为：
   $$
   c_i= \begin{cases}k, & \text { 若 } x_i \text { 为聚类中心, 且归属于第 } k \text {个类 } ; \\ -1, & 若x_i\text{为异常点. }\end{cases}
   $$
   ​		初始化数组`c`为全零。

   ​		使用`cn`保存当前为第几个cluster。并按照定义对聚类中心和异常点的$c_i$进行赋值。

   

   （6）对剩余数据点进行归类

   ```python
   for i in range(self.m):
       if self.c[self.index[i]] == 0:
           j = np.argmin(self.d[self.index[i]][self.index[:i]])
           self.c[self.index[i]] = self.c[self.index[j]]
   ```

   ​		论文中的归类方法：After the cluster centers have been found, each remaining point is assigned to the same cluster as its nearest neighbor of higher density.

   ​		按照 $\rho$ 值从大到小的顺序进行遍历，逐层扩充每一个cluster，按照定义对剩余数据点的$c_i$进行赋值。

   

3. 画出决策图，选择样本中心和异常点

   ```python
   from matplotlib import pyplot as plt
   def draw_decision_graph(self):
       plt.scatter(self.rho, self.delta,alpha=0.5)
       plt.hlines(self.thr_delta,0,np.max(self.rho),linestyles='--',colors="lightcoral")
       plt.vlines(self.thr_rho,0,np.max(self.delta),linestyles='--',colors='lightcoral')
       plt.xlabel("rho")
       plt.ylabel("delta")
       plt.title(f'决策图\n dc={self.dc}, thr_rho={self.thr_rho}, thr_delta={self.thr_delta}')
       plt.show()
       print("聚类中心：")
       for i in self.center:
           print("rho:",self.rho[i]," delta:",self.delta[i],"数据点:",self.X[i])
       print("异常点：")
       print("rho: ",self.rho[np.where(self.c==-1)]," delta: ",self.delta[np.where(self.c==-1)], "数据点:", self.X[np.where(self.c==-1)])
   ```

   ​		调用 pyplot 库对决策图进行可视化。

   ​		人工根据决策图选择聚类中心和异常点，即选择$thr\_rho、thr\_delta$。

   ​		打印聚类中心和异常点。

   

4. 确定分簇结果，计算评价指标（DBI），画出可视化图

   ```python
   def plot_result(self):
       plt.scatter(x=self.X[:, 0], y=self.X[:, 1], c=self.c,cmap='tab20',alpha=0.4)
       plt.scatter(x=self.X[self.center][:, 0], y=self.X[self.center][:, 1], marker="x", s=50, c="r")
       plt.title(f'分簇结果\n dc={self.dc}, thr_rho={self.thr_rho}, thr_delta={self.thr_delta}')
       plt.show()
   ```

   ```python
   from sklearn.metrics import davies_bouldin_score as dbs
   print('DBI得分为：', dbs(dfA, dpc1.c))
   ```

   ​		调用 pyplot 库对分簇结果进行可视化，调用sklearn.metrics. davies_bouldin_score计算DBI指数。

   ​		DBI表示聚类之间的平均“相似度”，其中相似度是将聚类之间的距离与聚类本身的大小进行比较的度量，越小越好。



## 六、实验结果与分析

1. 数据集 Aggregation.txt

- 原始数据

<img src="D:\桌面文件\image-20221211122559348.png" alt="image-20221211122559348" style="zoom: 80%;" />

- 调参

  ​		取`d`中`i<j`的部分，并对其进行升序排序，取前1%~2%位置的数，1%处为0.0434，2%处为0.0620。

  | kernel   | dc     | thr_rho | thr_delta | DBI    |
  | -------- | ------ | ------- | --------- | ------ |
  | cut-off  | 0.0434 | 9       | 0.2       | 0.5461 |
  | Gaussian | 0.0434 | 4       | 0.2       | 0.6570 |
  | cut-off  | 0.0620 | 15      | 0.19      | 0.5435 |
  | Gaussian | 0.0620 | 6       | 0.18      | 0.5435 |
  | cut-off  | 0.08   | 20      | 0.18      | 0.7141 |
  | Gaussian | 0.08   | 10      | 0.14      | 0.5435 |

<img src="D:\桌面文件\image-20221211141438125.png" alt="image-20221211141438125" style="zoom: 50%;" /><img src="D:\桌面文件\image-20221211141532329.png" alt="image-20221211141532329" style="zoom:50%;" />

<img src="D:\桌面文件\image-20221211141841071.png" alt="image-20221211141841071" style="zoom:50%;" /><img src="D:\桌面文件\image-20221211142053899.png" alt="image-20221211142053899" style="zoom:50%;" />

<img src="D:\桌面文件\image-20221211142530488.png" alt="image-20221211142240423" style="zoom:50%;" /><img src="D:\桌面文件\image-20221211142240423.png" alt="image-20221211142240423" style="zoom:50%;" />

​				cut-off kernel 为离散值，Gaussian kernel 为连续值。后者产生冲突的概率更小。

​				可见取 $kernel = Gaussian,d_c=0.08$ 时，聚类中心较明显，且DBI较低。



- 最终聚类结果

  - 决策图

    ![image-20221211142728060](D:\桌面文件\image-20221211142728060.png)

    ​		根据决策图，取$thr\_rho = 10，thr\_delta = 0.14$。得到：

    <img src="D:\桌面文件\image-20221211121407175.png" alt="image-20221211121407175" style="zoom: 50%;" />

    ​		共7个cluster，没有异常点。

    

  - 聚类结果图

  ![image-20221211183523514](D:\桌面文件\image-20221211183523514.png)

  

  - DBI

    <img src="D:\桌面文件\image-20221211142956934.png" alt="image-20221211142956934" style="zoom:67%;" />

    

2. 数据集 D31.txt

- 原始数据

  <img src="D:\桌面文件\image-20221211183533924.png" alt="image-20221211183533924" style="zoom:80%;" />

- 调参

  ​		取`d`中`i<j`的部分，并对其进行升序排序，取前1%~2%位置的数，1%处为0.0346，2%处为0.0542。

  | kernel   | dc   | thr_rho | thr_delta | DBI    |
  | -------- | ---- | ------- | --------- | ------ |
  | cut-off  | 0.04 | 35      | 0.07      | 0.5527 |
  | Gaussian | 0.04 | 15      | 0.065     | 0.5527 |
  | cut-off  | 0.05 | 40      | 0.07      | 0.5525 |
  | Gaussian | 0.05 | 20      | 0.065     | 0.5520 |

  ​		4种参数的决策图差异不大，取$kernel = Gaussian,d_c=0.05$时，DBI较低。

  

- 最终聚类结果
  - 决策图

    ![image-20221211144047296](D:\桌面文件\image-20221211144047296.png)

    ​		根据决策图，取$thr\_rho = 20，thr\_delta = 0.065$。得到：

    <img src="D:\桌面文件\image-20221211130014538.png" alt="image-20221211130014538" style="zoom: 33%;" />

    ​		共31个cluster，没有异常点。

    

  - 聚类结果图

    ![image-20221211183602682](D:\桌面文件\image-20221211183602682.png)

    

  - DBI

    <img src="D:\桌面文件\image-20221211130745694.png" alt="image-20221211130745694" style="zoom: 67%;" />



3. 数据集 R15.txt

- 原始数据

  <img src="D:\桌面文件\image-20221211125301248.png" alt="image-20221211125301248" style="zoom:80%;" />

- 调参

  | kernel   | dc    | thr_rho | thr_delta | DBI    |
  | -------- | ----- | ------- | --------- | ------ |
  | cut-off  | 0.02  | 10      | 0.07      | 0.3147 |
  | Gaussian | 0.02  | 4       | 0.07      | 0.3147 |
  | cut-off  | 0.025 | 12      | 0.07      | 0.3147 |
  | Gaussian | 0.025 | 5       | 0.07      | 0.3147 |

  ​		取`d`中`i<j`的部分，并对其进行升序排序，取前1%~2%位置的数，1%处为0.0181，2%处为0.0268。几种参数决策图都较好，DBI相同。任意选一种参数$kernel = cut-off, dc=0.02$。

  

- 最终聚类结果

  - 决策图

  ![image-20221211145150366](D:\桌面文件\image-20221211145150366.png)

  ​			根据决策图，取$thr\_rho = 6，thr\_delta = 0.07$。得到：

  <img src="D:\桌面文件\image-20221211145218355.png" alt="image-20221211145218355" style="zoom:40%;" />

  ​			共15个cluster，没有异常点。

  

  - 聚类结果图

    ![image-20221211183615975](D:\桌面文件\image-20221211183615975.png)

    

  - DBI

    <img src="D:\桌面文件\image-20221211134638988.png" alt="image-20221211134638988" style="zoom:67%;" />

