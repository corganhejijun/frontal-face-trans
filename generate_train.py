# -*- coding: utf-8 -*- 
import dlib
import os
import numpy as np
import cv2
from src.face_landmark import getFaceDis, getStandardFace
from src.util import getFace, getBound, transFaceImg
from src.resize_for_train import resizeX2, combineImg

FRONT_FACE_STANDARD = "datasets/eigen_face.jpg"
SHAPE_MODEL = "models/shape_predictor_68_face_landmarks.dat"
DATASET_DIR = "datasets/celeba_train"
DEST_DIR = "datasets/celeba_train_ab"

shapePredict = dlib.shape_predictor(SHAPE_MODEL)
detector = dlib.get_frontal_face_detector()
FRONT_THRESHOLD_DISTANCE = 30
ext = '.jpg'
IMAGE_SIZE = 256
standardLandmarks = getStandardFace(FRONT_FACE_STANDARD, detector, shapePredict)

class Face:
    def __init__(self, img, shape, detect, fileName):
        self.img = img
        self.shape = shape
        self.detect = detect
        self.fileName = fileName

if (not os.path.exists(DEST_DIR)):
    os.mkdir(DEST_DIR)
    
number = 0
counter = 0
folderList = os.listdir(DATASET_DIR)
for subFolder in folderList:
    counter += 1
    print("processing subFolder %s, %.2f%% complete" % (subFolder, (counter / len(folderList) * 100)))
    frontList = []
    otherList = []
    for fileName in os.listdir(os.path.join(DATASET_DIR, subFolder)):
        if not fileName.endswith(ext):
            continue
        filePath = os.path.join(DATASET_DIR, subFolder, fileName)
        img = cv2.cvtColor(cv2.imread(filePath), cv2.COLOR_BGR2RGB)
        landmarks, detect, shape = getFaceDis(img, detector, shapePredict, fileName)
        face = Face(img, shape, detect, fileName)
        if landmarks is None:
            continue
        diff = np.linalg.norm(landmarks - standardLandmarks)
        if (diff < FRONT_THRESHOLD_DISTANCE):
            frontList.append(face)
        else:
            otherList.append(face)
    print("subFolder {0} has {1} front faces".format(subFolder, len(frontList)))
    for frontFace in frontList:
        frontXmin, _, frontYmin, _ = getBound(frontFace.img, frontFace.shape)
        ctrlDstPts = np.zeros((frontFace.shape.num_parts,2))
        front, frontMargin = resizeX2(frontFace, IMAGE_SIZE)
        for i in range(frontFace.shape.num_parts):
            ctrlDstPts[i] = [frontFace.shape.part(i).x - frontXmin, 
                                frontFace.shape.part(i).y - frontYmin]
        for face in otherList:
            trans = transFaceImg(face.detect, face.shape, face.img, ctrlDstPts, face.fileName)
            if np.any(trans == None):
                continue
            other, otherMargin = resizeX2(trans, IMAGE_SIZE)
            result = combineImg(front, other)
            number += 1
            result.save(os.path.join(DEST_DIR, str(number).zfill(6) + ext))