# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from scipy import misc
from shutil import copyfile

LFW_ALIGH = "result/lfw"
FRONT_DIR = "result/face_frontal_lfw"
DEST_DIR = "result/frontal_align"
DEST_DIR_FULL = "result/frontal_align_full"

if not os.path.exists(DEST_DIR):
    os.mkdir(DEST_DIR)
if not os.path.exists(DEST_DIR_FULL):
    os.mkdir(DEST_DIR_FULL)
dirList = os.listdir(LFW_ALIGH)
for index, dirName in enumerate(dirList):
    print("processing {0}, {1} out of total {2}".format(dirName, index, len(dirList)))
    subDir = os.path.join(LFW_ALIGH, dirName)
    ffPath = os.path.join(FRONT_DIR, dirName)
    for file in os.listdir(subDir):
        ffFile = os.path.join(ffPath, file)
        if not os.path.exists(os.path.join(DEST_DIR_FULL, dirName)):
            os.mkdir(os.path.join(DEST_DIR_FULL, dirName))
        if not os.path.exists(ffFile):
            copyfile(os.path.join(LFW_ALIGH, dirName, file), os.path.join(DEST_DIR_FULL, dirName, file))
            continue
        if not os.path.exists(os.path.join(DEST_DIR, dirName)):
            os.mkdir(os.path.join(DEST_DIR, dirName))
        img = cv2.cvtColor(cv2.imread(ffFile), cv2.COLOR_BGR2RGB)
        resizeImg = cv2.resize(img, (200, int(200*img.shape[0]/img.shape[1])))
        resultImg = np.zeros((256, 256, 3))
        b = int((resizeImg.shape[0] - 200) / 2)
        resultImg[28:228, 28:228] = resizeImg[b:b+200,:]
        misc.imsave(os.path.join(DEST_DIR, dirName, file), resultImg)
        misc.imsave(os.path.join(DEST_DIR_FULL, dirName, file), resultImg)
        