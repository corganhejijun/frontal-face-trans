# -*- coding: utf-8 -*-
import dlib
import os
import numpy as np 
import cv2
from src.face_landmark import getFaceDis, getStandardFace

folder = "datasets/celeba/celeba_identified"
FRONT_FACE_STANDARD = "datasets/eigen_face.jpg"
SHAPE_MODEL = "models/shape_predictor_68_face_landmarks.dat"
DEST_FILE = "datasets/frontal_list.txt"

shapePredict = dlib.shape_predictor(SHAPE_MODEL)
detector = dlib.get_frontal_face_detector()
FRONT_THRESHOLD_DISTANCE = 50
MINIMUM_FACE_SIZE = 50

ext = '.jpg'
standardLandmarks = getStandardFace(FRONT_FACE_STANDARD, detector, shapePredict)
file = open(DEST_FILE, 'a')
counter = 0
folderList = os.listdir(folder)
for subFolder in folderList:
    counter += 1
    fileList = os.listdir(os.path.join(folder, subFolder))
    
    for fileName in fileList:
        if not fileName.endswith(ext):
            continue
        srcFilePath = os.path.join(folder, subFolder, fileName)
        try:
            img = cv2.cvtColor(cv2.imread(srcFilePath), cv2.COLOR_BGR2RGB)
        except Exception as err:
            print("{0} folder file {1} read failed, err:{2}.".format(subFolder, fileName, str(err)))
            continue
        landmarks, _, shape = getFaceDis(img, detector, shapePredict, fileName)
        if landmarks is None:
            continue
        diff = np.linalg.norm(landmarks - standardLandmarks)
        print("%s %s %.2f" % (fileName, subFolder, diff))
        if (diff < FRONT_THRESHOLD_DISTANCE):
            file.write(fileName + " " + subFolder + " " + str(diff) + "\n")
    percent = counter / len(folderList) * 100
    print("%s folder processed, %.2f%% complete" % (subFolder, percent))
