import numpy as np 
import cv2


vid=cv2.VideoCapture(0)

yuz_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')



while True :
    
    ret , frame = vid.read()
    
    gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    yuzler=yuz_cascade.detectMultiScale(gray,1.3,5)
    
    # x=> sağ sol 
    # y=> yukarı aşağı 
    
    # x , y , w, h 
    # 238 ,141, 212, 212 
    
    for (x,y,w,h) in yuzler :
        
        cv2.rectangle(frame,(x,y) ,(x+w ,y+h),(85,255,0),3)
    cv2.imshow("title",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
vid.release()
cv2.destroyAllWindows()