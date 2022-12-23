# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 18:36:07 2022

@author: safak
"""
import cv2

face_detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap=cv2.VideoCapture(0)

while True:
    ret, frame=cap.read()
    
    if ret:
        face_rec=face_detector.detectMultiScale(frame,minNeighbors=10,scaleFactor=1.02)
        
        for (x,y,w,h) in face_rec:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255),2)
        cv2.imshow("face detection",frame)
        
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()