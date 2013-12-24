import cv2
import numpy as np

def sobel(image):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    img = image
    img = cv2.GaussianBlur(img,(3,3),0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Gradient-X
    grad_x = cv2.Sobel(gray,ddepth,1,0,ksize = 3, scale = scale, delta = delta,borderType = cv2.BORDER_DEFAULT)
    #grad_x = cv2.Scharr(gray,ddepth,1,0)

    # Gradient-Y
    grad_y = cv2.Sobel(gray,ddepth,0,1,ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
    #grad_y = cv2.Scharr(gray,ddepth,0,1)

    abs_grad_x = cv2.convertScaleAbs(grad_x)   # converting back to uint8
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    dst = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
    #dst = cv2.add(abs_grad_x,abs_grad_y)

    cv2.imshow('dst',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=="__main__":

    XML_PATH = '/home/kaushal/Documents/Vedio2Text/jake/external/OpenCV-2.4.3/data/haarcascades/'
    FACESET = XML_PATH + 'haarcascade_frontalface_default.xml'
    EYESET = XML_PATH + 'haarcascade_eye.xml'
    MOUTHSET = XML_PATH + 'haarcascade_mcs_mouth.xml'
    DOWNSCALE = 4
     
    webcam = cv2.VideoCapture('../capture.avi')
    cv2.namedWindow("preview")
    classifier1 = cv2.CascadeClassifier(FACESET)
    classifier2 = cv2.CascadeClassifier(EYESET)
    classifier3 = cv2.CascadeClassifier(MOUTHSET)
    print webcam.isOpened()

    if webcam.isOpened(): # try to get the first frame
        rval, frame = webcam.read()
    else:
        rval = False
    print rval 
    while rval:
     
        # detect faces and draw bounding boxes
        minisize = (frame.shape[1]/DOWNSCALE,frame.shape[0]/DOWNSCALE)
        miniframe = cv2.resize(frame, minisize)
        #gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = classifier1.detectMultiScale(miniframe)
        for f in faces:
            x, y, w, h = [ v*DOWNSCALE for v in f ]
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))
            roi_gray = frame[y:y+h,x:x+w]
            #roi_color = img[y:y+h, x:x+w]
            eyes=classifier2.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,255,0))
            moi_gray = frame[y+h/2:y+h,x:x+w]
            mouth=classifier3.detectMultiScale(moi_gray)
            for (mx,my,mw,mh) in mouth:
                cv2.rectangle(moi_gray,(mx,my),(mx+mw,my+mh),(255,0,0))
        #sobel(moi_gray[my:my+mh,mx:mx+mw])


     
        cv2.putText(frame, "Press ESC to close.", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
        cv2.imshow("preview", frame)
     
        # get next frame
        rval, frame = webcam.read()
     
        key = cv2.waitKey(20)
        if key in [27, ord('Q'), ord('q')]: # exit on ESC
            break
