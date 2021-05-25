import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

#haar cascade classifiers for detection of face, eyes 
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl=['Close', 'Open']        #The eyes can be either labelled as Closed or Open 

model = load_model('models/cnncat3.h5')     #loading the model from models directory

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
count=0         #intial count and score are set to 0
score=0
thicc=2         #initial thickness is 2px
rpred=[-1]      #eye prediction values are set to -1
lpred=[-1]



mixer.init()     #alarm sounds
sound1 = mixer.Sound('alarms/alarm1.wav')
sound2 = mixer.Sound('alarms/alarm2.wav')


while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray_image, minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
    left_eye = leye.detectMultiScale(gray_image)
    right_eye =  reye.detectMultiScale(gray_image)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (100,100,100), 1 )

    cv2.rectangle(frame, (700,0) , (450,50) , (255,255,255) , thickness=cv2.FILLED )
    cv2.rectangle(frame, (0,height-50), (150,height), (255,0,0), thickness=cv2.FILLED )

    for (x,y,w,h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]
        count = count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye = r_eye/255
        r_eye = r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict_classes(r_eye)
        if(rpred[0] == 1):
            lbl='Open' 
        if(rpred[0] == 0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]
        count = count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye = l_eye/255
        l_eye = l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0] == 1):
            lbl='Open'   
        if(lpred[0] == 0):
            lbl='Closed'
        break

    if(rpred[0]==0 and lpred[0]==0):
        score = score+1
        cv2.putText(frame,"Eyes Closed!!",(500,30), font, 1,(0,0,255),1,cv2.LINE_AA)
    else:       #if(rpred[0]==1 or lpred[0]==1)
        score=score-1
        cv2.putText(frame,"Eyes Open..",(500,30), font, 1,(0,255,0),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(30,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score<5):
        sound1.stop()
        sound2.stop()
    if(score>10):
        path = os.getcwd()       
        cv2.imwrite(os.path.join(path,'screenshots/image.jpg'),frame)   #taking screenshot
        try:
            #ringing the alarm, as the person is sleepy
            sound1.play()
            sound2.play()          
        except:
            pass

        #increasing thickness while the score is increasing
        if(thicc<20):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)

    #exit keys to stop the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#destroying all windows
cap.release()
cv2.destroyAllWindows()
