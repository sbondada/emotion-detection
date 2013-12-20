#!/usr/bin/env python

import sys, os
import cv2
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


def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes 

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    # resize to given size (if given)
                    if (sz is not None):
                        im = im.resize(sz, Image.ANTIALIAS)
                    im=np.asarray(im, dtype=np.uint8)
                    X.append(extractItem(im,'face'))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c+1
    return [X,y]

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
    # This is where we write the images, if an output_dir is given
    # in command line:
    out_dir = None
    # You'll need at least a path to your image data, please see
    # the tutorial coming with this source code on how to prepare
    # your image data:
    if len(sys.argv) < 2:
        print "USAGE: facerec_demo.py </path/to/images>"
        sys.exit()
    # Now read in the image data. This must be a valid path!
    size=640,490
    [X,y] = read_images(sys.argv[1],size)
    print X[4].shape
    print y
