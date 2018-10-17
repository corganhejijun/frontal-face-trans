
# -*- coding: utf-8 -*- 
from src.file_oper import copy_celeba
import dlib
import cv2
import numpy as np
import os
from shutil import copyfile

CELEBA_IDENTITY_FILE = "datasets\\celeba\\identity_CelebA.txt"
CELEBA_DATASET = "datasets\\celeba\\img_align_celeba"
MIDDLE_DIR = "datasets\\celeba\\celeba_identified"

copy_celeba(CELEBA_IDENTITY_FILE, CELEBA_DATASET, MIDDLE_DIR)

FRONT_FACE_STANDARD = "datasets\\eigen_face.jpg"
SHAPE_MODEL = "models\\shape_predictor_68_face_landmarks.dat"
DEST_DIR = "datasets\\celeba_train"

shapePredict = dlib.shape_predictor(SHAPE_MODEL)
detector = dlib.get_frontal_face_detector()
NOSE_CENTER_NUMBER = 30
FRONT_THRESHOLD_DISTANCE = 30

def getStandardFace():
    img = cv2.cvtColor(cv2.imread(FRONT_FACE_STANDARD), cv2.COLOR_BGR2RGB)
    dets = detector(img, 1)
    shape = shapePredict(img, dets[0])
    landmarkList = np.zeros((shape.num_parts, 2))
    noseCenter = shape.part(NOSE_CENTER_NUMBER)
    for i in range(shape.num_parts):
        landmarkList[i] = [shape.part(i).x - noseCenter.x, shape.part(i).y - noseCenter.y]
    return landmarkList

def getFaceDis(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    dets = detector(img, 1)
    if (len(dets) != 1):
        print("Face number is {0} for {1}, detect failed".format(len(dets), path))
        return None
    detect = dets[0]
    if (detect.left() < 0):
        print("Face detect left is minus for {0}".format(path))
        return None
    if (detect.right() < 0):
        print("Face detect right is minus for {0}".format(path))
        return None
    if (detect.top() < 0):
        print("Face detect top is minus for {0}".format(path))
        return None
    if (detect.bottom() < 0):
        print("Face detect bottom minus for {0}".format(path))
        return None
    shape = shapePredict(img, detect)
    landmarkList = np.zeros((shape.num_parts, 2))
    noseCenter = shape.part(NOSE_CENTER_NUMBER)
    for i in range(shape.num_parts):
        landmarkList[i] = [shape.part(i).x - noseCenter.x, shape.part(i).y - noseCenter.y]
    return landmarkList
    
folder = MIDDLE_DIR
ext = '.jpg'
standardLandmarks = getStandardFace()
if (not os.path.exists(DEST_DIR)):
    os.mkdir(DEST_DIR)
for subFolder in os.listdir(folder): 
    fileList = os.listdir(os.path.join(folder, subFolder))
    if len(fileList) < 2:
        print("{0} has less than 2 files".format(subFolder))
        continue
    frontalFile = ""
    for fileName in fileList:
        if not fileName.endswith(ext):
            continue
        srcFilePath = os.path.join(folder, subFolder, fileName)
        destFilePath = os.path.join(DEST_DIR, subFolder, fileName)
        if os.path.exists(destFilePath):
            continue
        landmarks = getFaceDis(srcFilePath)
        if landmarks is None:
            continue
        diff = np.linalg.norm(landmarks - standardLandmarks)
        if (diff < FRONT_THRESHOLD_DISTANCE):
            if not os.path.exists(os.path.join(DEST_DIR, subFolder)):
                os.mkdir(os.path.join(DEST_DIR, subFolder))
            copyfile(srcFilePath, destFilePath)
            frontalFile = fileName
            print("{0} folder has frontal file {1}".format(subFolder, fileName))
            break
    if len(frontalFile) == 0:
        print("{0} folder don't have frontal file".format(subFolder))
        continue
    for fileName in fileList:
        if frontalFile != fileName:
            srcFilePath = os.path.join(folder, subFolder, fileName)
            destFilePath = os.path.join(DEST_DIR, subFolder, fileName)
            copyfile(srcFilePath, destFilePath)
    print("{0} folder copyed".format(subFolder))
