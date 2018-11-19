# -*- coding: utf-8 -*- 
import dlib
import os
import numpy as np
import cv2
from src.face_landmark import getFaceDis, getStandardFace
from src.util import getFace, getBound, transFaceImg
from src.resize_for_train import resizeMargin, combineImg

FRONT_FACE_STANDARD = "datasets/eigen_face.jpg"
SHAPE_MODEL = "models/shape_predictor_68_face_landmarks.dat"
DATASET_DIR = "datasets/celeba_train"
DEST_DIR = "datasets/celeba_train_on_mask"

shapePredict = dlib.shape_predictor(SHAPE_MODEL)
detector = dlib.get_frontal_face_detector()
FRONT_THRESHOLD_DISTANCE = 30
ext = '.jpg'
IMAGE_SIZE = 128
BLACK_POINT_VALUE = 100
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
    print("processing subFolder %s, current %d of %d complete" % (subFolder, counter, len(folderList)))
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
        xmin, xmax, ymin, ymax = getBound(frontFace.img, frontFace.shape)
        ctrlDstPts = np.zeros((frontFace.shape.num_parts,2))
        front, frontMargin = resizeMargin(frontFace.img[ymin:ymax,xmin:xmax,:], IMAGE_SIZE)
        for i in range(frontFace.shape.num_parts):
            ctrlDstPts[i] = [frontFace.shape.part(i).x - xmin, 
                                frontFace.shape.part(i).y - ymin]
        for face in otherList:
            trans = transFaceImg(face.detect, face.shape, face.img, ctrlDstPts, face.fileName)
            if np.any(trans == None):
                continue
            other, otherMargin = resizeMargin(trans, IMAGE_SIZE)
            frontWithMsk = np.copy(front)
            for i in range(len(other)):
                for j in range(len(other[0])):
                    if np.sum(other[i][j]) < BLACK_POINT_VALUE:
                        frontWithMsk[i][j] = other[i][j]
            result = combineImg(front, frontWithMsk)
            number += 1
            result.save(os.path.join(DEST_DIR, str(number).zfill(6) + ext))