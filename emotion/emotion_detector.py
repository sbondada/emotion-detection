#!/usr/bin/env python

import cv2
import sys, os
sys.path.append("../..")
# import facerec modules
from facerec.feature import Fisherfaces, SpatialHistogram, Identity
from facerec.distance import EuclideanDistance, ChiSquareDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
from facerec.visual import subplot
from facerec.util import minmax_normalize
from facerec.serialization import save_model, load_model
# import numpy, matplotlib and logging
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from facerec.lbp import LPQ, ExtendedLBP

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
        faceCropImage=cv2.resize(inputImage[y:y+h,x:x+w],(340,340))
    if item == 'eye':
        faceCropImage=cv2.resize(inputImage,(340,340))
    return faceCropImage
 
if __name__ == "__main__":
    im = Image.open('/home/kaushal/Dropbox/github/emotion-detection/666/Image3.png')
    im = im.convert("L")
    #resize to given size (if given)
    sz=640,490
    if (sz is not None):
       im = im.resize(sz, Image.ANTIALIAS)
    X=extractItem(np.asarray(im, dtype=np.uint8),'face')

    model = load_model('model.pkl')
    # Then turn the first (at most) 16 eigenvectors into grayscale
    # images (note: eigenvectors are stored by column!)
    print model.predict(X) 
