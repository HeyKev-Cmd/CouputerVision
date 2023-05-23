# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 19:35:56 2020

@author: kevin
"""

import matplotlib.pyplot as plt
import cv2
# 載入分類器
face_cascade = cv2.CascadeClassifier('C:/Users/kevin/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
# 讀取圖片
img = cv2.imread(r'D:/kevin/code/data_science/CouputerVision/face.jpeg')
path=r'fast&furious6.jpg'
# img = cv2.imread(path)
# 轉成灰階圖片

# cv2.imshow('Image',img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#%%%
# 偵測臉部


faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=3,
    minSize=(32, 32))

# 参数1：image--待检测图片，一般为灰度图像加快检测速度；

# 参数2：objects--被检测物体的矩形框向量组；
# 参数3：scaleFactor--表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%;
# 参数4：minNeighbors--表示构成检测目标的相邻矩形的最小个数(默认为3个)。
#         如果组成检测目标的小矩形的个数和小于 min_neighbors - 1 都会被排除。
#         如果min_neighbors 为 0, 则函数不做任何操作就返回所有的被检候选矩形框，
#         这种设定值一般用在用户自定义对检测结果的组合程序上；
# 参数5：flags--要么使用默认值，要么使用CV_HAAR_DO_CANNY_PRUNING，如果设置为

#         CV_HAAR_DO_CANNY_PRUNING，那么函数将会使用Canny边缘检测来排除边缘过多或过少的区域，

#         因此这些区域通常不会是人脸所在区域；
# 参数6、7：minSize和maxSize用来限制得到的目标区域的范围。


#%%
# 繪製人臉部份的方框
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#(0, 255, 0)欄位可以變更方框顏色(Blue,Green,Red)
# cv2.rectangle(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度)
#%%
# 顯示成果
cv2.namedWindow('img', cv2.WINDOW_NORMAL)  #正常視窗大小
cv2.imshow('img', img)                     #秀出圖片
# cv2.imwrite( "result.jpg", img )           #保存圖片
cv2.waitKey(0)                             #等待按下任一按鍵
cv2.destroyAllWindows()       