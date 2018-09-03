# -*- coding: utf-8 -*- 
import cv2
import os
from scipy import misc
import numpy as np
import dlib
import math

from .MovingLSQ import MovingLSQ
from .FaceDistance import distance

def getBound(img, shape):
    xMin = len(img[0])
    xMax = 0
    yMin = len(img)
    yMax = 0
    for i in range(shape.num_parts):
        if (shape.part(i).x < xMin):
            xMin = shape.part(i).x
        if (shape.part(i).x > xMax):
            xMax = shape.part(i).x
        if (shape.part(i).y < yMin):
            yMin = shape.part(i).y
        if (shape.part(i).y > yMax):
            yMax = shape.part(i).y
    return xMin, xMax, yMin, yMax

def getFace(detector, shapePredict, file):
    img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    dets = detector(img, 1)
    if (len(dets) == 0):
        print("file %s has no face" % file)
        return None
    det = dets[0]
    shape = shapePredict(img, det)
    xmin, xmax, ymin, ymax = getBound(img, shape)
    if xmin < 0 or xmax < 0 or ymin < 0 or ymax < 0:
        print("file %s can't get bound" % file)
        return None
    return img[ymin:ymax,xmin:xmax,:]

def transFace(detector, shapePredict, file, controlDstPts, fullSize):
    img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    dets = detector(img, 1)
    if (len(dets) == 0):
        print("file %s has no face" % file)
        return None
    det = dets[0]
    shape = shapePredict(img, det)
    xmin, xmax, ymin, ymax = getBound(img, shape)
    if xmin < 0 or xmax < 0 or ymin < 0 or ymax < 0:
        print("file %s can't get bound" % file)
        return None
    cropImg = img[ymin:ymax,xmin:xmax,:]
    controlSrcPts = np.zeros((shape.num_parts,2))
    for i in range(shape.num_parts):
        if (shape.part(i).x < 0):
            print("%d th part x < 0 for %s" % (i, file))
            return None
        if (shape.part(i).y < 0):
            print("%d th part y < 0 for %s" % (i, file))
            return None
        controlSrcPts[i] = [shape.part(i).x - xmin, shape.part(i).y - ymin]
    solver = MovingLSQ(controlSrcPts, controlDstPts)
    imgIdx = np.zeros(((xmax - xmin)*(ymax - ymin), 2))
    for i in range((ymax - ymin)):
        for j in range((xmax - xmin)):
            imgIdx[i*(xmax - xmin) + j] = [j, i]
    imgMls = solver.Run_Rigid(imgIdx)
    mlsMargin = [65536, 65536, -65536, -65536]
    for i in range(len(imgMls)):
        if (imgMls[i][0] < mlsMargin[0]):
            mlsMargin[0] = imgMls[i][0]
        if (imgMls[i][1] < mlsMargin[1]):
            mlsMargin[1] = imgMls[i][1]
        if (imgMls[i][0] > mlsMargin[2]):
            mlsMargin[2] = imgMls[i][0]
        if (imgMls[i][1] > mlsMargin[3]):
            mlsMargin[3] = imgMls[i][1]
    mlsMargin[2] -= (xmax - xmin)
    mlsMargin[3] -= (ymax - ymin)
    imgMlsMap = imgMls.reshape(((ymax - ymin), (xmax - xmin), 2))
    leftMargin = -math.floor(mlsMargin[0])
    topMargin = -math.floor(mlsMargin[1])
    rightMargin = math.ceil(mlsMargin[2])
    bottomMargin = math.ceil(mlsMargin[3])
    deformedImage = np.zeros(((ymax - ymin) + int(topMargin) + int(bottomMargin), 
                                (xmax - xmin) + int(leftMargin) + int(rightMargin), 3))
    for i in range(len(cropImg)):
        for j in range(len(cropImg[0])):
            x = int(math.floor(imgMlsMap[i][j][0]) + leftMargin)
            y = int(math.floor(imgMlsMap[i][j][1]) + topMargin)
            if (x < 0 or y < 0):
                break
            deformedImage[y, x] = cropImg[i, j]
    return deformedImage 


def faceFromDir(inDir, outDir, shape_model):
    shapePredict = dlib.shape_predictor(shape_model)
    detector = dlib.get_frontal_face_detector()
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    for dirName in os.listdir(inDir):
        print("processing %s" % dirName)
        subDir = os.path.join(inDir, dirName)
        if os.path.isdir(subDir):
            imgList = []
            fileNameList = []
            for imgFile in os.listdir(subDir):
                if not imgFile.endswith('.jpg'):
                    continue
                img = getFace(detector, shapePredict, os.path.join(subDir, imgFile))
                if np.any(img == None):
                    continue
                imgList.append(img)
                fileNameList.append(imgFile)
            if len(imgList) < 2:
                continue
            outPath = os.path.join(outDir, dirName);
            if not os.path.exists(outPath):
                os.mkdir(outPath)
            for index, img in enumerate(imgList):
                misc.imsave(os.path.join(outPath, fileNameList[index]), img)

def transFromDir(inDir, aligned_data, outDir, shape_model, eigenPath, fullSize):
    standardImg = cv2.cvtColor(cv2.imread(eigenPath), cv2.COLOR_BGR2RGB)
    detector = dlib.get_frontal_face_detector()
    shapePredict = dlib.shape_predictor(shape_model)
    standardDets = detector(standardImg, 1)[0]
    standardShape = shapePredict(standardImg, standardDets)
    stanXmin, _, stanYmin, _ = getBound(standardImg, standardShape)
    controlDstPts = np.zeros((standardShape.num_parts,2))
    for i in range(standardShape.num_parts):
        controlDstPts[i] = [standardShape.part(i).x - stanXmin, standardShape.part(i).y - stanYmin]

    if not os.path.exists(outDir):
        os.mkdir(outDir)
    for dirName in os.listdir(inDir):
        print("processing %s" % dirName)
        subDir = os.path.join(inDir, dirName)
        if os.path.isdir(subDir):
            imgList = []
            fileNameList = []
            for imgFile in os.listdir(subDir):
                if not imgFile.endswith('.jpg'):
                    continue
                if not os.path.exists(os.path.join(aligned_data, dirName, imgFile)):
                    continue
                img = transFace(detector, shapePredict, os.path.join(subDir, imgFile),
                                controlDstPts, fullSize)
                if np.any(img == None):
                    if os.path.exists(os.path.join(aligned_data, dirName, imgFile)):
                        print(imgFile + " exists but not processed")
                    continue
                imgList.append(img)
                fileNameList.append(imgFile)
            if len(imgList) < 2:
                continue
            outPath = os.path.join(outDir, dirName);
            if not os.path.exists(outPath):
                os.mkdir(outPath)
            for index, img in enumerate(imgList):
                misc.imsave(os.path.join(outPath, fileNameList[index]), img)

def getDatasetDistance(result_path, dataset_path, model_path):
    file = open(result_path, "w")
    file.write("name, average, standard, count\n")
    file.close()
    for dirName in os.listdir(dataset_path):
        print("calculating distance of dir %s" % dirName)
        subDir = os.path.join(dataset_path, dirName)
        avg, std = distance(subDir, model_path, 160)
        file = open(result_path, "a")
        file.write("%s, %f, %f, %d\n" % (dirName, avg, std, len(os.listdir(subDir))))
        file.close()