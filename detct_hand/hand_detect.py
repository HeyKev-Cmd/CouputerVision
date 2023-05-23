# -*- coding: utf-8 -*-
"""
Created on Sun May 23 10:43:56 2021

@author: kevin
"""
import cv2
import mediapipe as mp
import time
cap=cv2.VideoCapture(0)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils
cTime=0
pTime=0
while True: 
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            # for id, lm in enumerate(hand.landmark):
            #     print(id,lm)
            #     h,w,c=img.shape
            #     cx,cy=int(lm.x*w), int(lm.y*h)
            #     pass
            mpDraw.draw_landmarks(img,hand,mpHands.HAND_CONNECTIONS)
            
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)        
    cv2.imshow('image',img)
    cv2.waitKey(1)
    