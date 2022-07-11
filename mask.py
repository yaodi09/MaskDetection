import cv2 as cv
from keras.backend import maximum
import numpy as np
import pandas as pd

from keras.preprocessing.image import img_to_array
from keras.models import load_model


model = load_model("./cp-10.h5")

cam = cv.VideoCapture(0) 

cascade_path = './haarcascade_frontalface_default.xml'
haar = cv.CascadeClassifier(cascade_path)

#live camera test
'''
while True:
    (check, frame) = cam.read()
    frame = cv.flip(frame, 1, 1)

    cv.imshow('Mask Detector', frame)
    key = cv.waitKey(1)
    if key == 27:
        break
'''

on_off = {0:'NO MASK ON', 1:'MASK ON'}
rect_color = {0:(0, 0 ,255), 1:(0, 255, 0)}
font = cv.FONT_HERSHEY_DUPLEX
font_color = {0:(0, 0 ,255), 1:(0, 255, 0)}

while True:
    (check, frame) = cam.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #print(frame.shape[1], frame.shape[0])
    #720
    height = frame.shape[0] // 4
    #1280
    width = frame.shape[1] // 4

    frame_resize = cv.resize(frame, (width, height))
    faces = haar.detectMultiScale(frame_resize)
    #faces = haar.detectMultiScale(frame)
    for (x, y, w, h) in faces:
        x, y, w, h = x*4, y*4, w*4, h*4
        #cv.putText(frame, 'TEXT HERE', (x, y-15), font, 1, (1, 1, 1), 1)
        #cv.rectangle(frame, (x, y), (x+w, y+h), (1, 1 ,1), 4)
        face_img = frame[y:y+h, x:x+w]
        face_img = cv.resize(face_img, (150, 150))
        face_img = face_img/255.0
        face_img = np.reshape(face_img, (1, 150, 150, 3))
        face_img = np.vstack([face_img])
        pred = model.predict(face_img)
        #print(pred)
        #pred_max = np.where(pred[0] == maximum)
        pred_max = np.argmax(pred, axis = 1, out = None)[0]
        #print(pred_max)

        print(on_off[pred_max])

        cv.putText(frame, on_off[pred_max], (x, y-15), font, 1, font_color[pred_max], 1)
        cv.rectangle(frame, (x, y), (x+w, y+h), rect_color[pred_max], 4)

    cv.imshow('Mask Detector', frame)
    
    if cv.waitKey(1) == 27: 
        break

cam.release()
cv.destroyAllWindows()