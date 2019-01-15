# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from scipy import misc

SRC_DATA = "datasets/lfw_aligned"
SRC_DATA_FULL = "compare/result/lfw_aligned_full"
DEST_DIR = "compare/result/lfw"
MARGIN = 28
IN_SIZE = 200
SIZE = 256

if not os.path.exists(DEST_DIR):
    os.mkdir(DEST_DIR)
dirList = os.listdir(SRC_DATA_FULL)
for index, dirName in enumerate(dirList):
    print("processing {0}, {1} out of total {2}".format(dirName, index, len(dirList)))
    for fullfile in os.listdir(os.path.join(SRC_DATA_FULL, dirName)):
        file = fullfile[:-4] + ".jpg"
        if not os.path.exists(os.path.join(SRC_DATA, dirName, file)):
            fullFile = os.path.join(SRC_DATA_FULL, dirName, fullfile)
            fullImg = cv2.cvtColor(cv2.imread(fullFile), cv2.COLOR_BGR2RGB)
            if not os.path.exists(os.path.join(DEST_DIR, dirName)):
                os.mkdir(os.path.join(DEST_DIR, dirName))
            misc.imsave(os.path.join(DEST_DIR, dirName, file), fullImg)
            continue
        filePath = os.path.join(SRC_DATA, dirName, file)
        img = cv2.cvtColor(cv2.imread(filePath), cv2.COLOR_BGR2RGB)
        resultImg = np.zeros((SIZE, SIZE, 3))
        if img.shape[0] > img.shape[1]:
            resizeImg = cv2.resize(img, (int(IN_SIZE*img.shape[1]/img.shape[0]), IN_SIZE))
        else:
            resizeImg = cv2.resize(img, (IN_SIZE, int(IN_SIZE*img.shape[0]/img.shape[1])))
        resultImg[MARGIN:MARGIN+resizeImg.shape[0], MARGIN:MARGIN+resizeImg.shape[1]] = resizeImg
        resultDir = os.path.join(DEST_DIR, dirName)
        if not os.path.exists(resultDir):
            os.mkdir(resultDir)
        misc.imsave(os.path.join(resultDir, file), resultImg)