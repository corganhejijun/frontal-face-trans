# -*- coding: utf-8 -*- 
import cv2
import os
from scipy import misc
import numpy as np
import dlib

def getBound(img, shape):
    xMin = len(img[0])
    xMax = 0
    yMin = len(img)
    yMax = 0
    for i in range(shape.num_parts):
        if (shape.part(i).x < xMin):
            xMin = shape.part(i).x
        if (shape.part(i).x > xMax):
            xMax = shape.part(i).x
        if (shape.part(i).y < yMin):
            yMin = shape.part(i).y
        if (shape.part(i).y > yMax):
            yMax = shape.part(i).y
    return xMin, xMax, yMin, yMax

def getFace(detector, shapePredict, file):
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

def faceFromDir(inDir, outDir, shape_model):
    shapePredict = dlib.shape_predictor(shape_model)
    detector = dlib.get_frontal_face_detector()
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    for dirName in os.listdir(inDir):
        print("processing %s" % dirName)
        subDir = os.path.join(inDir, dirName)
        if os.path.isdir(subDir):
            imgList = []
            fileNameList = []
            for imgFile in os.listdir(subDir):
                if not imgFile.endswith('.jpg'):
                    continue
                img = getFace(detector, shapePredict, os.path.join(subDir, imgFile))
                if np.any(img == None):
                    continue
                imgList.append(img)
                fileNameList.append(imgFile)
            if len(imgList) < 2:
                continue
            outPath = os.path.join(outDir, dirName);
            if not os.path.exists(outPath):
                os.mkdir(outPath)
            for index, img in enumerate(imgList):
                misc.imsave(os.path.join(outPath, fileNameList[index]), img)