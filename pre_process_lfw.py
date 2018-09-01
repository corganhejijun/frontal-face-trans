# -*- coding: utf-8 -*- 
import os
import cv2
from scipy import misc
import dlib
from src.util import getBound
import numpy as np

IN_DIR = "datasets\\lfw"
OUT_DIR = "datasets\\lfw_aligned"

shapePredict = dlib.shape_predictor("models\\shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

def getFace(file):
    img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    dets = detector(img, 1)
    if (len(dets) == 0):
        print("file %s has no face" % file)
        return None
    det = dets[0]
    shape = shapePredict(img, det)
    xmin, xmax, ymin, ymax = getBound(img, shape)
    if xmin < 0 or xmax < 0 or ymin < 0 or ymax < 0:
        print("file %s can't get bound" % file)
        return None
    return img[ymin:ymax,xmin:xmax,:]

if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
for dirName in os.listdir(IN_DIR):
    print("processing %s" % dirName)
    subDir = os.path.join(IN_DIR, dirName)
    if os.path.isdir(subDir):
        imgList = []
        fileNameList = []
        for imgFile in os.listdir(subDir):
            if not imgFile.endswith('.jpg'):
                continue
            img = getFace(os.path.join(subDir, imgFile))
            if np.any(img == None):
                continue
            imgList.append(img)
            fileNameList.append(imgFile)
        if len(imgList) < 2:
            continue
        outPath = os.path.join(OUT_DIR, dirName);
        if not os.path.exists(outPath):
            os.mkdir(outPath)
        for index, img in enumerate(imgList):
            misc.imsave(os.path.join(outPath, fileNameList[index]), img)
        
            