
# -*- coding: utf-8 -*- 
from src.file_oper import copy_celeba
import dlib
import os
import numpy as np
from shutil import copyfile
from src.face_landmark import getFaceDis, getStandardFace

CELEBA_IDENTITY_FILE = "datasets/celeba/identity_CelebA.txt"
CELEBA_DATASET = "datasets/celeba/img_align_celeba"
MIDDLE_DIR = "datasets/celeba/celeba_identified"

copy_celeba(CELEBA_IDENTITY_FILE, CELEBA_DATASET, MIDDLE_DIR)

FRONT_FACE_STANDARD = "datasets/eigen_face.jpg"
SHAPE_MODEL = "models/shape_predictor_68_face_landmarks.dat"
DEST_DIR = "datasets/celeba_train"

shapePredict = dlib.shape_predictor(SHAPE_MODEL)
detector = dlib.get_frontal_face_detector()
FRONT_THRESHOLD_DISTANCE = 30

folder = MIDDLE_DIR
ext = '.jpg'
standardLandmarks = getStandardFace(FRONT_FACE_STANDARD, detector, shapePredict)
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
        landmarks = getFaceDis(srcFilePath, detector, shapePredict)
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
