import cv2
import numpy as np

def sobel(inputImage):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    img = cv2.GaussianBlur(inputImage,(3,3),0)
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
    #conveting the image in gray scale to black and white using the threshold
    #thresh = 25
    #dst = cv2.threshold(dst, thresh, 255, cv2.THRESH_BINARY)[1]
    return dst


def extractItem(inputImage,item):
    XML_PATH = '/home/kaushal/Documents/Vedio2Text/jake/external/OpenCV-2.4.3/data/haarcascades/'
    if item == 'face':
        ITEMSET = XML_PATH + 'haarcascade_frontalface_default.xml'
    elif item == 'mouth':
        ITEMSET = XML_PATH + 'haarcascade_mcs_mouth.xml'
        h,w,d=inputImage.shape
        #cutting the image to concentrate only on the lower part of the face
        inputImage=inputImage[h/2:h,:w]
    elif item == 'eye':
        ITEMSET = XML_PATH + 'haarcascade_eye.xml'
        h,w,d=inputImage.shape
        #cutting the image to concentrate only on the lower part of the face
        inputImage=inputImage[:h/2,:w]

    DOWNSCALE = 4
    classifier = cv2.CascadeClassifier(ITEMSET)

    # detect faces 
    minisize = (inputImage.shape[1]/DOWNSCALE,inputImage.shape[0]/DOWNSCALE)
    miniframe = cv2.resize(inputImage, minisize)
    items = classifier.detectMultiScale(miniframe)
    for i in items:
        x, y, w, h = [ v*DOWNSCALE for v in i ]
        if item =='face':
            cv2.rectangle(inputImage, (x,y), (x+w,y+h), (0,0,255))
        if item == 'mouth':
            cv2.rectangle(inputImage, (x,y), (x+w,y+h), (0,255,0))
        if item == 'eye':
            cv2.rectangle(inputImage, (x,y), (x+w,y+h), (255,0,0))
        
    #extract the face from the image to return the face
    if item == 'face' or item == 'mouth':
        faceCropImage=inputImage[y:y+h,x:x+w]
    if item == 'eye':
        faceCropImage=inputImage
    return faceCropImage
 
'''
#code which extracts the mouth and applies sobel filter and displays the image in black and white
if __name__=="__main__":
    DATASET_PATH='/home/kaushal/Documents/emotion_database/'
    inputImage=cv2.imread(DATASET_PATH+'cohn-kanade-images/S005/001/S005_001_00000010.png')
    cv2.namedWindow("preview")
    faceCropImage=extractItem(inputImage,'face')
    mouthCropImage=extractItem(faceCropImage,'mouth')
    sobelMouth=sobel(mouthCropImage)
    #applying haris corner detection algorithm
    harisMouth= cv2.cornerHarris(sobelMouth,2,3,0.04)
    mouthCropImage[harisMouth>0.01*harisMouth.max()]=[0,0,255]
    #making the corner points viewable 
    cv2.imshow("preview", mouthCropImage)
    key = cv2.waitKey(2000)

'''
'''
#code for extracting swift points and siplaying them
if __name__=="__main__":
    DATASET_PATH='/home/kaushal/Documents/emotion_database/'
    inputImage=cv2.imread(DATASET_PATH+'cohn-kanade-images/S005/001/S005_001_00000010.png')
    cv2.namedWindow("preview")
    faceCropImage=extractItem(inputImage,'face')
    mouthCropImage=extractItem(faceCropImage,'mouth')
    sobelMouth=sobel(mouthCropImage)
    #applying sift 
    sift=cv2.SIFT()
    kp, des = sift.detectAndCompute(mouthCropImage,None)
    #displaying the key point in sift
    img=cv2.drawKeypoints(mouthCropImage,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("preview",img)
    key = cv2.waitKey(2000)
'''
if __name__=="__main__":
    DATASET_PATH='/home/kaushal/Documents/emotion_database/'
    inputImage=cv2.imread(DATASET_PATH+'cohn-kanade-images/S005/001/S005_001_00000010.png')
    cv2.namedWindow("preview")
    faceCropImage=extractItem(inputImage,'face')
    mouthCropImage=extractItem(faceCropImage,'mouth')
    sobelMouth=sobel(mouthCropImage)
    cv2.imshow("preview",sobelMouth)
    key = cv2.waitKey(6000)
    
