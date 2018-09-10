# -*- coding: utf-8 -*- 
import os
from scipy import misc
from PIL import Image
import numpy as np
import math

def resizeX2(img, fullSize):
    margin = [0, 0, 0, 0] # [top, left, bottom, right]
    imgX2 = np.zeros((img.shape[0]*2, img.shape[1]*2, img.shape[2]))
    for i in range(len(imgX2)):
        for j in range(len(imgX2[0])):
            imgX2[i][j] = img[int(i/2)][int(j/2)]
    if imgX2.shape[0] > fullSize:
        imgX2 = imgX2[:fullSize,:,:]
    if imgX2.shape[1] > fullSize:
        imgX2 = imgX2[:,:fullSize,:]
    fullsizeImg = np.zeros((fullSize, fullSize, 3))
    left = int((fullSize - imgX2.shape[1])/2)
    top = int((fullSize - imgX2.shape[0])/2)
    fullsizeImg[top:imgX2.shape[0]+top, left:imgX2.shape[1]+left, :] = imgX2
    margin[2] = margin[0] = top
    margin[3] = margin[1] = left
    return fullsizeImg, margin

def duplicateImg(img):
    target = Image.new('RGB', (img.shape[0]*2, img.shape[1]))
    target.paste(Image.fromarray(np.uint8(img)), (0, 0))
    target.paste(Image.fromarray(np.uint8(img)), (img.shape[0] + 1, 0))
    return target

def getTrainImg(folder, fullSize, outDir, marginFile):
    file = open(marginFile, "w")
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
    for subFolder in os.listdir(folder):
        subFolderPath = os.path.join(folder, subFolder)
        if not os.path.isdir(subFolderPath):
            continue
        for imgFile in os.listdir(subFolderPath):
            if not imgFile.endswith('.jpg'):
                continue
            print("processing {0}".format(imgFile))
            imgPath = os.path.join(subFolderPath, imgFile)
            img = misc.imread(imgPath)
            imgX2, margin = resizeX2(img, fullSize)
            img4Train = duplicateImg(imgX2)
            img4Train.save(os.path.join(outDir, imgFile))
            file.write("{0}:{1}\n".format(imgFile, margin))
    file.close()