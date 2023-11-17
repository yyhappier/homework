# C++ 课前热身练习实验报告

*袁雨 PB20151804*



## 一、实验目的

- 学习使用 CMake 来搭建项目
- 学习使用 Visual Studio 2019 进行编程，学会其 debug 工具来调试代码
- 学习面向对象 C++ 编程，特别是类（ `class` ）的封装特性及构造函数、析构函数、函数重载、运算符重载等
- 熟悉 C++ 指针、动态内存分配、预编译头机制等
- 学习模板 `template` 
- 学习STL的 `vector`、`list`、 `map` 等
- 学习静态库 lib，动态库 dll 的编写



## 二、实验内容

### 小练习 1. 基础的动态数组

> 详细说明见于 [documents/1_BasicDArray](../documents/1_BasicDArray) 

完成 [src/executables/1_BasicDArray](src/executables/1_BasicDArray) 

### 小练习 2. 高效的动态数组

> 详细说明文档见于 [documents/2_EfficientDArray](../documents/2_EfficientDArray) 

完成 [src/executables/2_EfficientDArray](src/executables/2_EfficientDArray) 

### 小练习 3. 模板动态数组

> 详细说明见于 [documents/3_TemplateDArray](../documents/3_TemplateDArray) 

仿照小练习 1，在文件夹 [src/executables/](src/executables) 中添加文件夹 `3_TemplateDArray`，并在其内

- 添加文件 `TemplateDArray.h` 
- 添加文件 `main.cpp` 
- 添加文件 `CMakeLists.txt`，同于 [src/executables/1_BasicDArray/CMakeLists.txt](src/executables/1_BasicDArray/CMakeLists.txt) 

重新 CMake 后得到新子项目 3_TemplateDArray

### 小练习 4. 基于 `list` 的多项式类

> 详细说明见于 [documents/4_list_Polynomial](../documents/4_list_Polynomial) 

这里将 PolynomialList 编写成了动态库，具体查看 [src/libraries/shared](src/libraries/shared)，编译后会生成 [lib/](lib)CppPractices_libraries_shared(d).dll

你需要补充完成 [src/libraries/shared/PolynomialList.cpp](src/libraries/shared/PolynomialList.cpp) 

[4_list_Polynomial](src/executables/4_list_Polynomial) 会测试该动态库

###  小练习 5. 基于 `map` 的多项式类

> 详细说明见于 [documents/5_map_Polynomial](../documents/5_map_Polynomial) 

这里将 PolynomialMap 编写成了静态库，具体查看 [src/libraries/static](src/libraries/static)，编译后会生成 [lib/](lib)CppPractices_libraries_static(d).dll

你需要补充完成 [src/libraries/static/PolynomialMap.cpp](src/libraries/static/PolynomialMap.cpp) 

[5_map_Polynomial](src/executables/5_map_Polynomial) 会测试该静态库，另外该子项目还用到了小练习 4 的动态库 PolynomialList，其中会测试小练习 4 和小练习 5 的性能差异



## 三、实验结果

### 小练习 1. 基础的动态数组

<img src="C:\Users\袁雨\AppData\Roaming\Typora\typora-user-images\image-20230305195029204.png" alt="image-20230305195029204" style="zoom: 40%;" />



### 小练习 2. 高效的动态数组

<img src="C:\Users\袁雨\AppData\Roaming\Typora\typora-user-images\image-20230305195120401.png" alt="image-20230305195120401" style="zoom:40%;" />



### 小练习 3. 模板动态数组

<img src="C:\Users\袁雨\AppData\Roaming\Typora\typora-user-images\image-20230305195501981.png" alt="image-20230305195501981" style="zoom:40%;" />



### 小练习 4. 基于 `list` 的多项式类

<img src="C:\Users\袁雨\AppData\Roaming\Typora\typora-user-images\image-20230305195610885.png" alt="image-20230305195610885" style="zoom:45%;" />



### 小练习 5. 基于 `map` 的多项式类

![image-20230305212703000](D:\桌面文件\image-20230305212703000.png)



## 四、实验分析与心得体会

​		通过小练习1，了解和掌握了 C++ 的基本语法。初步了解类（对象）的编写，了解构造函数、析构函数、函数重载等；熟悉和巩固了指针、动态内存分配的机制与操作。

​		通过小练习2，学会了使用预先多分配一些内存的方法来提高动态数组的效率，理解对象的 `public `接口的重要性。

​		通过小练习3，学会了使用 `template`来处理不同类型的动态数组类，初步了解和使用`STL`的`vector`。

​		通过小练习4， 学会了使用 `vector` 和 `list`、了解生成和使用动态库。

​		通过小练习5， 学会了使用 `map` 来实现 `polynomial` 类，了解静态库的编写与使用；

​		对比发现，`list<T>`容器是双向链表，因此可以有效的在任何位置添加和删除。列表的缺点是不能随机访问内容，要想访问内容必须在列表的内部从头开始便利内容，或者从尾部开始。`map<K, T>`映射容器：K表示键，T表示对象，根据特定的键映射到对象，可以进行快速的检索。`map`的效率更高，二者的时间复杂度不同。

