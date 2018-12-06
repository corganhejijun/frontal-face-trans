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
DATASET_DIR = "datasets/celeba_train_hd"
DEST_DIR = "datasets/celeba_train_on_mask_256"

shapePredict = dlib.shape_predictor(SHAPE_MODEL)
detector = dlib.get_frontal_face_detector()
FRONT_THRESHOLD_DISTANCE = 30
ext = '.jpg'
IMAGE_SIZE = 256
BLACK_POINT_VALUE = 100
FACE_WIDTH_ADD = 5
FACE_BOTTOM_ADD = 5
FACE_TOP_ADD = 15
TRANS_SCALE = 2
standardLandmarks = getStandardFace(FRONT_FACE_STANDARD, detector, shapePredict)

class Face:
    def __init__(self, img, trans_img, shape, detect, fileName):
        self.img = img
        self.trans_img = trans_img
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
        trans_img = cv2.resize(img, (int(img.shape[1] / TRANS_SCALE), int(img.shape[0] / TRANS_SCALE)))
        landmarks, detect, shape = getFaceDis(trans_img, detector, shapePredict, fileName)
        face = Face(img, trans_img, shape, detect, fileName)
        if landmarks is None:
            continue
        diff = np.linalg.norm(landmarks - standardLandmarks)
        if (diff < FRONT_THRESHOLD_DISTANCE):
            frontList.append(face)
        else:
            otherList.append(face)
    print("subFolder {0} has {1} front faces".format(subFolder, len(frontList)))
    for frontFace in frontList:
        xmin, xmax, ymin, ymax = getBound(frontFace.trans_img, frontFace.shape)
        ctrlDstPts = np.zeros((frontFace.shape.num_parts,2))
        for i in range(frontFace.shape.num_parts):
            ctrlDstPts[i] = [frontFace.shape.part(i).x - xmin, 
                                frontFace.shape.part(i).y - ymin]
        if ymin * TRANS_SCALE - FACE_TOP_ADD >= 0:
            ymin = ymin * TRANS_SCALE - FACE_TOP_ADD
        if ymax * TRANS_SCALE + FACE_BOTTOM_ADD <= frontFace.img.shape[0]:
            ymax = ymax * TRANS_SCALE + FACE_BOTTOM_ADD
        if xmin * TRANS_SCALE - FACE_WIDTH_ADD >= 0:
            xmin = xmin * TRANS_SCALE - FACE_WIDTH_ADD
        if xmax * TRANS_SCALE + FACE_WIDTH_ADD <= frontFace.img.shape[1]:
            xmax = xmax * TRANS_SCALE + FACE_WIDTH_ADD
        front, _ = resizeMargin(frontFace.img[ymin:ymax,xmin:xmax,:], IMAGE_SIZE)
        for face in otherList:
            try:
                trans = transFaceImg(face.detect, face.shape, face.trans_img, ctrlDstPts, face.fileName)
            except Exception as err:
                print("file {0} read failed, err:{1}.".format(face.fileName, str(err)))
                continue
            if np.any(trans == None):
                continue
            other, _ = resizeMargin(trans, int(IMAGE_SIZE / TRANS_SCALE))
            frontWithMsk = np.copy(front)
            for i in range(len(other)):
                for j in range(len(other[0])):
                    if np.sum(other[i][j]) < BLACK_POINT_VALUE:
                        for k in range(TRANS_SCALE):
                            frontWithMsk[i*TRANS_SCALE + k][j*TRANS_SCALE : (j+1)*TRANS_SCALE] = other[i][j]
            result = combineImg(front, frontWithMsk)
            number += 1
            result.save(os.path.join(DEST_DIR, face.fileName))