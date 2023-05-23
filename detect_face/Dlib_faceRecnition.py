# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 10:11:32 2020

@author: kevin
"""


import dlib
import cv2
import imutils
# img = cv2.imread(r'D:/kevin/code/data_science/CouputerVision/face.jpeg')
path=r'face.jpeg'
# path=r'CIMG4105.JPG'
img = cv2.imread(path)
# 讀取圖檔
# img = cv2.imread('image.jpg')

# 縮小圖片
img = imutils.resize(img, width=1280)

# Dlib 的人臉偵測器
detector = dlib.get_frontal_face_detector()

# 偵測人臉
# face_rects = detector(img, 0)
face_rects, scores, idx = detector.run(img, 0,-1)
# 第二參數代表将原始图像是否进行放大，1表示放大1倍再检查，提高小人脸的检测效果
# score分數越大，說明detector更確信是人臉
# detector.run中第三個參數是可選擇的檢測閾值
    # 檢測閾值為負，將會返回更多的檢測結果
    # 檢測閾值為正，將會返回較少的檢測結果
# idx將告訴我們是哪個子檢測器與圖像中第i張人臉匹配。
    # 這將更廣泛地用於識別不同方向的人臉
# 取出所有偵測的結果
for i, d in enumerate(face_rects):
  x1 = d.left()
  y1 = d.top()
  x2 = d.right()
  y2 = d.bottom()
  text = "%2.2f(%d)" % (scores[i], idx[i])

# 以方框標示偵測的人臉
  cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
  cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,
          0.7, (255, 255, 255), 1, cv2.LINE_AA)
# 顯示結果
cv2.imshow("Face Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()