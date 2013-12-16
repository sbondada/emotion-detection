import numpy as np
import cv2

#create black image

img = np.zeros((512,512,3),np.uint8)

#Draw a blue diagonal line with thikness of 5 px
img = cv2.line(img, (0,0), (511,511), (255,0,0), 5)

#Draw a green rectangle - only need to provide top left and bottom right of rect to draw
img = cv2.rectangle(img, (384,0), (510,128), (0,255,0), 3)

#Draw a red circle - provide center and radius
img = cv2.circle(img, (447,63), 63, (0,0,255),-1)

#put text on image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'Hey Hoo', (10,500), font, (255,255,255), 2,thickness=4, linetype=cv2.CV_AA)

cv2.imshow('draw on image', img)
cv2.waitKey(5)
cv2.destroyAllWindows()

