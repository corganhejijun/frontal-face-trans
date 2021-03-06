
# -*- coding: utf-8 -*- 
from src.file_oper import copy_celeba
import dlib
import os
import numpy as np 
import cv2
from shutil import copyfile
from src.face_landmark import getFaceDis, getStandardFace
from src.util import getBound

CELEBA_IDENTITY_FILE = "datasets/celeba/identity_CelebA.txt"
CELEBA_DATASET = "datasets/celeba/img_celeba"
MIDDLE_DIR = "datasets/celeba/celeba_identified"

copy_celeba(CELEBA_IDENTITY_FILE, CELEBA_DATASET, MIDDLE_DIR)

FRONT_FACE_STANDARD = "datasets/eigen_face.jpg"
SHAPE_MODEL = "models/shape_predictor_68_face_landmarks.dat"
DEST_DIR = "datasets/celeba_train_hd"

shapePredict = dlib.shape_predictor(SHAPE_MODEL)
detector = dlib.get_frontal_face_detector()
FRONT_THRESHOLD_DISTANCE = 50
MINIMUM_FACE_SIZE = 50

folder = MIDDLE_DIR
ext = '.jpg'
standardLandmarks = getStandardFace(FRONT_FACE_STANDARD, detector, shapePredict)
if (not os.path.exists(DEST_DIR)):
    os.mkdir(DEST_DIR)
counter = 0
folderList = os.listdir(folder)
for subFolder in folderList:
    if os.path.exists(os.path.join(DEST_DIR, subFolder)):
        print("{0} folder exists".format(subFolder))
        continue
    counter += 1
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
        try:
            img = cv2.cvtColor(cv2.imread(srcFilePath), cv2.COLOR_BGR2RGB)
        except Exception as err:
            print("{0} folder file {1} read failed, err:{2}.".format(subFolder, fileName, str(err)))
            continue
        landmarks, _, shape = getFaceDis(img, detector, shapePredict, fileName)
        if landmarks is None:
            continue
        xmin, xmax, ymin, ymax = getBound(img, shape)
        if xmax - xmin < MINIMUM_FACE_SIZE and ymax - ymin < MINIMUM_FACE_SIZE:
            print("{0} folder file {1} has face size of {2} x {3}, too small.".format(subFolder, fileName, 
                                                                                        (xmax - xmin), (ymax - ymin)))
            continue
        diff = np.linalg.norm(landmarks - standardLandmarks)
        if (diff < FRONT_THRESHOLD_DISTANCE):
            if not os.path.exists(os.path.join(DEST_DIR, subFolder)):
                os.mkdir(os.path.join(DEST_DIR, subFolder))
            copyfile(srcFilePath, destFilePath)
            frontalFile = fileName
            print("{0} folder has frontal file {1}".format(subFolder, fileName))
            break
    percent = counter / len(folderList) * 100
    if len(frontalFile) == 0:
        print("%s folder don't have frontal file, %.2f%% complete" % (subFolder, percent))
        continue
    for fileName in fileList:
        if frontalFile != fileName:
            srcFilePath = os.path.join(folder, subFolder, fileName)
            destFilePath = os.path.join(DEST_DIR, subFolder, fileName)
            copyfile(srcFilePath, destFilePath)
    print("%s folder copyed, %.2f%% complete" % (subFolder, percent))
