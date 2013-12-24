import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    #Capture frame-by-frame
    ret, frame = cap.read()

    #Our operations on frame 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()

