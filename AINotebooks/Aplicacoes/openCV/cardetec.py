# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 14:48:26 2022

@author: cdieg
"""

import cv2
import time


# Criar o nosso classificador de carros
car_classifier = cv2.CascadeClassifier('haarcascades\haarcascade_car.xml')

# Iniciar uma captura de video
video_capture = cv2.VideoCapture('cars.avi')


# Loop once video is successfully loaded
while video_capture.isOpened():
    
    time.sleep(.05)
    # Primeiramente ler o frame
    ret, frame = video_capture.read()
    
    if not ret:
        break
    
    h, w, l = frame.shape
    resize = cv2.resize(frame, (w*2, h*2)) 
    cv2.imshow('Cars', resize) 
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
   
    # Passar para o classificador
    cars = car_classifier.detectMultiScale(gray, 1.2, 3)
    




    # Desenhar retângulos ao redor
    for (x,y,w,h) in cars:
        cv2.rectangle(resize, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Cars', resize)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()