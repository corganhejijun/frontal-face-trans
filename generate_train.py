# -*- coding: utf-8 -*- 
import dlib
import os
import numpy as np
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

if (not os.path.exists(DEST_DIR)):
    os.mkdir(DEST_DIR)
    
number = 0
for subFolder in os.listdir(DATASET_DIR): 
    print("processing subFolder {0}".format(subFolder))
    frontList = []
    otherList = []
    for fileName in os.listdir(os.path.join(DATASET_DIR, subFolder)):
        if not fileName.endswith(ext):
            continue
        filePath = os.path.join(DATASET_DIR, subFolder, fileName)
        landmarks = getFaceDis(filePath, detector, shapePredict)
        face = getFace(detector, shapePredict, filePath)
        if landmarks is None:
            continue
        diff = np.linalg.norm(landmarks - standardLandmarks)
        if (diff < FRONT_THRESHOLD_DISTANCE):
            frontList.append(face)
        else:
            otherList.append(face)
    print("subFolder {0} has {1} front faces".format(subFolder, len(frontList)))
    for frontFace in frontList:
        dets = detector(frontFace, 1)
        if len(dets) == 0:
            continue
        frontDets = dets[0]
        frontShape = shapePredict(frontFace, frontDets)
        frontXmin, _, frontYmin, _ = getBound(frontFace, frontShape)
        ctrlDstPts = np.zeros((frontShape.num_parts,2))
        front, frontMargin = resizeX2(frontFace, IMAGE_SIZE)
        for i in range(frontShape.num_parts):
            ctrlDstPts[i] = [frontShape.part(i).x - frontXmin, frontShape.part(i).y - frontYmin]
        for face in otherList:
            trans = transFaceImg(detector, shapePredict, face, ctrlDstPts, "")
            if np.any(trans == None):
                continue
            other, otherMargin = resizeX2(trans, IMAGE_SIZE)
            result = combineImg(front, other)
            number += 1
            result.save(os.path.join(DEST_DIR, str(number) + ext))