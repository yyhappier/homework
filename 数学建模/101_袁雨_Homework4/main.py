# -*- coding = utf-8 -*-
# @Author:袁雨
# @File:main.py
# @Software:PyCharm

import cv2
import numpy as np

# seamless cloning
# 读取图像文件
dest = cv2.imread("mg2.jpg")
source = cv2.imread("watermelon.png")
mask = cv2.imread("watermelon_mask.png", 0)

mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]
dest = cv2.resize(dest, (source.shape[1], source.shape[0]))

#3 channels for color image and 1 channel for grayscale image
if len(source.shape) == 2:
    source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
if len(dest.shape) == 2:
    dest = cv2.cvtColor(dest, cv2.COLOR_GRAY2BGR)
if len(mask.shape) < 3:
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# Normal clone
normal_clone = cv2.seamlessClone(
    source, dest, mask, (dest.shape[1]//2, dest.shape[0]//2), cv2.NORMAL_CLONE,blend=0.5)
# Mixed clone
mixed_clone = cv2.seamlessClone(
    source, dest, mask, (dest.shape[1]//2, dest.shape[0]//2), cv2.MIXED_CLONE,blend=0.5)
# Monochrome transfer
mon_clone = cv2.seamlessClone(
    source, dest, mask, (dest.shape[1]//2-20, dest.shape[0]//2+50), cv2.MONOCHROME_TRANSFER,blend=0.5)

cv2.imwrite("watermelon_normal_clone.jpg", normal_clone)
cv2.imwrite("watermelon_mixed_clone.jpg", mixed_clone)
cv2.imwrite("watermelon_mon_clone.jpg", mon_clone)


# local color changes

# 读取源图像文件
source= cv2.imread('flower2.jpg')

# 创建掩码，只对图像中间的一块进行颜色转换
mask = cv2.imread("flower2_mask.png", 0)

# 进行颜色转换
result = np.zeros(source.shape, source.dtype)
cv2.colorChange(source, mask, result, red_mul=1.5, green_mul=0.5, blue_mul=0.5)

# 显示原图和结果图
cv2.imwrite("flower2_color_red.jpg", result)


# local illumination change

# 读取源图像文件和掩码文件
source = cv2.imread('band.jpg')
mask = cv2.imread('band_mask.png', 0)  # 读取灰度掩码图像

# 创建结果图像矩阵
result =np.zeros_like(source)

# 设定增益系数
alpha = 100 # 控制亮度的增益系数，取值在 [0, +inf) 之间，1 为原图像
beta = 0.3# 控制对比度的增益系数，取值在 [0, +inf) 之间，1 为原图像

# 对图像进行亮度/对比度调整
cv2.illuminationChange(source, mask, result, alpha, beta)
cv2.imwrite("band_illu.jpg", result)


# Texture flattening

# 加载图像
source = cv2.imread("trump.jpg")

# 创建掩码
mask = cv2.imread("trump_mask (1).png", 0)

# 进行纹理平滑
result = cv2.textureFlattening(source, mask, low_threshold=20,high_threshold=90,kernel_size=3)

# 显示结果
cv2.imwrite("trump_flat.jpg", result)