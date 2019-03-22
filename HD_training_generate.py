# -*- coding: utf-8 -*- 
import os
import numpy as np
import cv2
import random
from src.resize_for_train import combineImg

OUT_SIZE = 256
NP_SAVE_DIR = "datasets/celeba_NP_MASK"
HD_CELEBA_DIR = "datasets/celeba/data_crop_512_jpg"
DEST_DIR = "datasets/HD_training_" + str(OUT_SIZE)
BLACK_POINT_VALUE = 100

counter = 0
folderList = os.listdir(HD_CELEBA_DIR)
npFileList = os.listdir(NP_SAVE_DIR)

if (not os.path.exists(DEST_DIR)):
    os.mkdir(DEST_DIR)

for file in folderList:
    counter += 1
    if counter % 10 == 0:
        print("current %d of %d complete" % (counter, len(folderList)))
    maskFile = random.sample(npFileList, 1)
    mask = np.load(os.path.join(NP_SAVE_DIR, maskFile[0]))
    xmin = 128
    xmax = 0
    ymin = 128
    ymax = 0
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if np.sum(mask[i][j]) > BLACK_POINT_VALUE:
                if i < ymin:
                    ymin = i
                if i > ymax:
                    ymax = i
                if j < xmin:
                    xmin = j
                if j > xmax:
                    xmax = j
    ymaxSize = ymax - ymin
    xmaxSize = xmax - xmin
    X_MASK_SCALE = OUT_SIZE / xmaxSize
    Y_MASK_SCALE = OUT_SIZE / ymaxSize
    img = cv2.cvtColor(cv2.imread(os.path.join(HD_CELEBA_DIR, file)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (OUT_SIZE, OUT_SIZE))
    imgMask = np.copy(img) 
    for i in range(ymin,ymax):
        for j in range(xmin,xmax):
            if np.sum(mask[i][j]) < BLACK_POINT_VALUE:
                for k in range(int(Y_MASK_SCALE)+1):
                    if (i-ymin)*Y_MASK_SCALE+k >= OUT_SIZE:
                        continue
                    if (j-xmin)*X_MASK_SCALE >= OUT_SIZE:
                        continue
                    end = int(((j-xmin)+1)*X_MASK_SCALE)
                    if end > OUT_SIZE:
                        end = OUT_SIZE
                    imgMask[int((i-ymin)*Y_MASK_SCALE+k)][int((j-xmin)*X_MASK_SCALE):end] = [0,0,0]
    result = combineImg(img, imgMask)
    result.save(os.path.join(DEST_DIR, file))
    

    
    