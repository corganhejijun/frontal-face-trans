# -*- coding: utf-8 -*- 
import cv2
import numpy as np
from src.util import getBound

NOSE_CENTER_NUMBER = 30

def getStandardFace(front_face, detector, shapePredict):
    img = cv2.cvtColor(cv2.imread(front_face), cv2.COLOR_BGR2RGB)
    dets = detector(img, 1)
    shape = shapePredict(img, dets[0])
    landmarkList = np.zeros((shape.num_parts, 2))
    noseCenter = shape.part(NOSE_CENTER_NUMBER)
    for i in range(shape.num_parts):
        landmarkList[i] = [shape.part(i).x - noseCenter.x, shape.part(i).y - noseCenter.y]
    return landmarkList

def getFaceDis(img, detector, shapePredict, path):
    dets = detector(img, 1)
    if (len(dets) != 1):
        print("Face number is {0} for {1}, detect failed".format(len(dets), path))
        return None, None, None
    detect = dets[0]
    if (detect.left() < 0):
        print("Face detect left is minus for {0}".format(path))
        return None, None, None
    if (detect.right() < 0):
        print("Face detect right is minus for {0}".format(path))
        return None, None, None
    if (detect.top() < 0):
        print("Face detect top is minus for {0}".format(path))
        return None, None, None
    if (detect.bottom() < 0):
        print("Face detect bottom minus for {0}".format(path))
        return None, None, None
    shape = shapePredict(img, detect)
    landmarkList = np.zeros((shape.num_parts, 2))
    noseCenter = shape.part(NOSE_CENTER_NUMBER)
    for i in range(shape.num_parts):
        landmarkList[i] = [shape.part(i).x - noseCenter.x, shape.part(i).y - noseCenter.y]
    return landmarkList, detect, shape

def resizeFace(img, detector, shape, path, size):
    landmarks, detect, shape = getFaceDis(img, detector, shape, path)
    if landmarks is None:
        return None
    if img.shape[0] < img.shape[1]:
        noseCenter = shape.part(NOSE_CENTER_NUMBER).x
        xmin = int(noseCenter - img.shape[0] / 2)
        xmax = int(noseCenter + img.shape[0] / 2)
        if xmin < 0:
            xmin = 0
            xmax = img.shape[0]
        if xmax > img.shape[1]:
            xmax = img.shape[1]
            xmin = xmax - img.shape[0]
        return cv2.resize(img[:, xmin:xmax, :], (size, size))
    noseCenter = shape.part(NOSE_CENTER_NUMBER).y
    ymin = int(noseCenter - img.shape[1] / 2)
    ymax = int(noseCenter + img.shape[1] / 2)
    if ymin < 0:
        ymin = 0
        ymax = img.shape[1]
    if ymax > img.shape[0]:
        ymax = img.shape[0]
        ymin = ymax - img.shape[1]
    return cv2.resize(img[ymin:ymax, :, :], (size, size))