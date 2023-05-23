# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:43:11 2021

@author: kevin
"""
import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
###
wCam,hCam=640,480
frameR=100 #frame reduction
smoothing=10

###
plocX,  plocY=0,0
clocX,  clocY=0,0

cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime=0
detector=htm.handDetector(maxHands=1)
wScr,hScr=autopy.screen.size()

while True:
    sucess,img=cap.read()
    # 1.find hand landmarks
    img=detector.findHands(img)
    lmList,bbox=detector.findPosition(img)
    
    # 2.get the tip of the index and middle fingers 
    if len(lmList)!=0:
        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:]
    # 3.check which finger is up 
        fingers=detector.fingersUp()
        cv2.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)
        # print(fingers)
    # 4.only index Finger
        if fingers[1]==1 and fingers[2]==0:
        # index finger true, middle fingure false
            # 5.convert coorridates
            x3=np.interp(x1,    (frameR,wCam-frameR),(0,wScr))
            y3=np.interp(y1,    (frameR,hCam-frameR),(0,hScr))
            cv2.circle(img,(x1,y1),15, (255,0,255),cv2.FILLED)
        # 6.soomthen values
            clocX=plocX+(x3-plocX)/smoothing
            clocY=plocY+(y3-plocY)/smoothing
            
            clocy=np.interp(x1,(frameR,hCam-frameR),(0,hScr))
        # 7.Move mouses
            autopy.mouse.move(wScr-clocX,clocY)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            plocX,plocY=clocX,clocY
        
    # 8. Both Index and middle finger up, clicking mode
        if fingers[1]==1 and fingers[2]==1:
            length,img,LineInfo=detector.findDistance(8,12,img)
        # print(length)
            if length<40:
                cv2.circle(img,(LineInfo[4],LineInfo[5]),15,(0,255,0),cv2.FILLED)
                autopy.mouse.click()
        
    # 9.find distance between fingers
    # 10.cick mouse if distance short
    # 11.frame rate
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
        (255, 0, 255), 3)
        
    # dispaly
    cv2.imshow('image',img)
    cv2.waitKey(1)