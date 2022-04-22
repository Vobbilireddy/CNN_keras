#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
mixer.init()
os.chdir("C:/Users/Anu/Downloads/Group_9 (1)/Group_9/Group_9")
music_to_be_played = mixer.Sound('alarm.wav')
frontal_face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
left_eye = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
right_eye = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
basic_cnn_model = load_model('drowsiness_detection.h5')
path = os.getcwd()
video_cap = cv2.VideoCapture(0)
font_name = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
var=2
right_pred=[99]
left_pred=[99]

while(True):
    rectangle, draw = video_cap.read()
    hei,wid = draw.shape[:2] 

    gray = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)
    
    faces = frontal_face.detectMultiScale(draw,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eyes = left_eye.detectMultiScale(draw)
    right_eyes =  right_eye.detectMultiScale(draw)

    cv2.rectangle(draw, (0,hei-50) , (200,hei) , (0,0,0) , thickness=cv2.FILLED )

    for (l,b,w,h) in faces:
        cv2.rectangle(draw, (l,b) , (l+w,b+h) , (100,100,100) , 1 )

    for (l,b,w,h) in right_eyes:
        right_eye_capture=draw[b:b+h,l:l+w]
        count=count+1
        right_eye_capture = cv2.cvtColor(right_eye_capture,cv2.COLOR_BGR2GRAY)
        right_eye_capture = cv2.resize(right_eye_capture,(256,256))
        right_eye_capture= right_eye_capture/255
        right_eye_capture=  right_eye_capture.reshape(256,256,-1)
        right_eye_capture = np.expand_dims(right_eye_capture,axis=0)
        right_pred_1 = basic_cnn_model.predict(right_eye_capture)
        print("*******************************************************************************")
        print(right_pred_1)
        class_labels=['Closed','Open']
        right_pred = class_labels[right_pred_1.argmax()]
        print("*******************************************************************************")
        print(right_pred)
        if(right_pred=='Open'):
            class_labels='Open' 
        if(right_pred=='Closed'):
            class_labels='Closed'
        break

    for (l,b,w,h) in left_eyes:
        left_eye_capture=draw[b:b+h,l:l+w]
        count=count+1
        left_eye_capture = cv2.cvtColor(left_eye_capture,cv2.COLOR_BGR2GRAY)  
        left_eye_capture = cv2.resize(left_eye_capture,(256,256))
        left_eye_capture= left_eye_capture/255
        left_eye_capture=left_eye_capture.reshape(256,256,-1)
        left_eye_capture = np.expand_dims(left_eye_capture,axis=0)
        left_pred_1 = basic_cnn_model.predict(left_eye_capture)

        print("----------------------------------------------------------------------------------")
        print(left_pred_1)
        class_labels=['Closed','Open']
        left_pred = class_labels[left_pred_1.argmax()]
        print("----------------------------------------------------------------------------------")
        print(left_pred)
        if(left_pred=='Open'):
            class_labels='Open' 
        if(left_pred=='Closed'):
            class_labels='Closed' 
        break
    if(right_pred=='Closed' and left_pred =='Closed'):
        score=score+1
        cv2.putText(draw,"Closed",(10,hei-20), font_name, 1,(255,255,255),1,cv2.LINE_AA)
    else:
        score=score-1
        cv2.putText(draw,"Open",(10,hei-20), font_name, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(draw,'Score:'+str(score),(100,hei-20), font_name, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>15):
        cv2.imwrite(os.path.join(path,'image.jpg'),draw)
        try:
            music_to_be_played.play()
        except: 
            pass
        if(var<16):
            var= var+2
        else:
            var=var-2
            if(var<2):
                var=2
        cv2.rectangle(draw,(0,0),(wid,hei),(0,0,255),var) 
    cv2.imshow('draw',draw)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




