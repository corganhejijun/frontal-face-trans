# -*- coding: utf-8 -*- 
import cv2
import numpy as np

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