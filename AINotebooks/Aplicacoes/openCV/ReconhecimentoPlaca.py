#!/usr/bin/env python
# coding: utf-8

# # Reconhecimento de placas em vídeo

# ## Importando as bibliotecas

# In[8]:


import cv2
import time

# ## Lendo o vídeo de entrada

# In[9]:


#############################################
frameWidth = 640
frameHeight = 480
nPlateCascade = cv2.CascadeClassifier("haarcascades/haarcascade_russian_plate_number.xml")
minArea = 200
color = (255,0,255)
###############################################
video_capture = cv2.VideoCapture("video12.mp4")
#cap.set(3, frameWidth)
#cap.set(4, frameHeight)
#cap.set(10,150)
count = 0


# ## Reconhecimento sistemático frame a frame

# In[ ]:


while (video_capture.isOpened()):
    time.sleep(.05)
    ret, frame = video_capture.read()
    
    if not ret :
        break
    
    h, w, l = frame.shape
    img = cv2.resize(cv2.rotate(frame, cv2.ROTATE_180), (w//4, h//4))
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 3)
    for (x, y, w, h) in numberPlates:
        area = w*h
        if area >minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img,"Placa",(x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
            imgRoi = img[y:y+h,x:x+w]
            cv2.imshow("Result", imgRoi)
 
    cv2.imshow("Result", img)
 
    pressedKey = cv2.waitKey(1) & 0xFF
    if pressedKey == ord('s'):
        cv2.imwrite("NoPlate_"+str(count)+".jpg",img)
        cv2.rectangle(img,(0,250),(500,300),(0,255,0),cv2.FILLED)
        cv2.putText(img,"Scan Saved",(50,280),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(0,0,255),2)
        cv2.imshow("Result", img)
        count +=1
        cv2.waitKey(0)
            
    if pressedKey == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows() 

# In[ ]:




