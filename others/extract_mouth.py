import cv2
 
#FACESET = "/home/kaushal/Documents/Vedio2Text/jake/external/OpenCV-2.4.3/data/lbpcascades/lbpcascade_frontalface.xml"
FACESET = "/home/kaushal/Documents/Vedio2Text/jake/external/OpenCV-2.4.3/data/haarcascades/haarcascade_frontalface_default.xml"
EYESET = "/home/kaushal/Documents/Vedio2Text/jake/external/OpenCV-2.4.3/data/haarcascades/haarcascade_eye.xml"
MOUTHSET = "/home/kaushal/Documents/Vedio2Text/jake/external/OpenCV-2.4.3/data/haarcascades/haarcascade_mcs_mouth.xml"
DOWNSCALE = 4
 
webcam = cv2.VideoCapture('../capture.avi')
cv2.namedWindow("preview")
classifier1 = cv2.CascadeClassifier(FACESET)
classifier2 = cv2.CascadeClassifier(EYESET)
classifier3 = cv2.CascadeClassifier(MOUTHSET)

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
            cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        moi_gray = frame[y+h/2:y+h,x:x+w]
        mouth=classifier3.detectMultiScale(moi_gray)
        for (mx,my,mw,mh) in mouth:
            cv2.rectangle(moi_gray,(mx,my),(mx+mw,my+mh),(255,0,0),2)
 

 
    cv2.putText(frame, "Press ESC to close.", (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
    cv2.imshow("preview", frame)
 
    # get next frame
    rval, frame = webcam.read()
 
    key = cv2.waitKey(20)
    if key in [27, ord('Q'), ord('q')]: # exit on ESC
        break
