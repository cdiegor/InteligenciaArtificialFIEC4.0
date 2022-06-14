# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:36:10 2022

@author: cdieg
"""

import cv2

faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier("haarcascades/haarcascade_smile.xml")
#bodyCascade = cv2.CascadeClassifier("haarcascades/haarcascade_fullbody.xml")
#ubodyCascade = cv2.CascadeClassifier("haarcascades/haarcascade_upperbody.xml")


video_capture = cv2.VideoCapture('http://10.100.10.95:4747/video')


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(15, 15),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    #bodies = bodyCascade.detectMultiScale(gray)
    
    #ubodies = ubodyCascade.detectMultiScale(gray)

    # Desenha um retângulo ao redor dos rostos
    #for (x, y, w, h) in bodies:
    #    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    #for (x, y, w, h) in ubodies:
    #    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Desenha um retângulo ao redor dos rostos
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.4, minNeighbors=2)
        # Desenha um novo retângulo ao redor dos sorrisos
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 255), 2)

    # Mostra o frame modificado
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()



