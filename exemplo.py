import numpy as np
import cv2 as cv
 
face_cascade = cv.CascadeClassifier('/media/faneli/B82A810D2A80CA38/bkp/Aulas/Data_Science/haarcascade_frontalface_default.xml')
 
#abrindo arquivo
img = cv.imread('pessoas5.jpg')
 
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
 
for (x,y,w,h) in faces:
  cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
  roi_gray = gray[y:y+h, x:x+w]
  roi_color = img[y:y+h, x:x+w]

#exibir caixa com a imagem gerada
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()