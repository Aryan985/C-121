import cv2
import time
import numpy as np
video = cv2.VideoWriter_fourcc(*"XVID")
output = cv2.VideoWriter("Output.avi",video,20.0,(640,480))
img = cv2.VideoCapture(0)
time.sleep(2)
bg = 0
for i in range(0,60):
    ret,bg = img.read()
bg = np.flip(bg,axis=1)
while(img.isOpened()):
    ret,frame = img.read()
    if not ret :
        break
    frame = np.flip(frame,axis=1)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lowred = np.array([0,120,50])
    highred = np.array([10,255,255])
    mask1 = cv2.inRange(hsv,lowred,highred)
    lowred = np.array([170,120,50])
    highred = np.array([180,255,255])
    mask2 = cv2.inRange(hsv,lowred,highred)
    mask1 = mask1+mask2
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
    mask2 = cv2.bitwise_not(mask1)
    res1 = cv2.bitwise_and(frame,frame,mask=mask2)
    res2 = cv2.bitwise_and(bg,bg,mask=mask1)
    finalres = cv2.addWeighted(res1,1,res2,1,0)
    output.write(finalres)
    cv2.imshow("Invisible",finalres)
img.release()
cv2.destroyAllWindows()