from .frontalize import ThreeD_Model, frontalize
from .facial_feature_detector import get_landmarks
from .camera_calibration import estimate_camera
import scipy.io as io
import cv2
import numpy as np
import os
from scipy import misc
import dlib

def frontalize_lfw(dataset, outDir):
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    predictor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "shape_predictor_68_face_landmarks.dat")
    modelPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontalization_models", "model3Ddlib.mat")
    model3D = ThreeD_Model(modelPath, 'model_dlib')
    eyemask = np.asarray(io.loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)) ,'frontalization_models', 'eyemask.mat'))['eyemask'])
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    dirList = os.listdir(dataset)
    for index, subDir in enumerate(dirList):
        print("processing {0}, {1} of total {2}".format(subDir, index, len(dirList)))
        subPath = os.path.join(dataset, subDir)
        for imgFile in os.listdir(subPath):
            imgPath = os.path.join(subPath, imgFile)
            img = cv2.imread(imgPath, 1)
            lmarks = get_landmarks(img, detector, predictor)
            try:
                proj_matrix, camera_matrix, rmat, tvec = estimate_camera(model3D, lmarks[0])
                eyemask = np.asarray(eyemask)
                frontal_raw, frontal_sym = frontalize(img, proj_matrix, model3D.ref_U, eyemask)
            except:
                print("{0} in {1} process error".format(imgFile, subDir))
                continue
            frontal = frontal_sym[:, :, ::-1]
            dets = detector(frontal, 1)
            if len(dets) < 1:
                continue
            det = dets[0]
            shape = predictor(frontal, det)
            xmin, xmax, ymin, ymax = getBound(img, shape)
            if not os.path.exists(os.path.join(outDir, subDir)):
                os.mkdir(os.path.join(outDir, subDir))
            outPath = os.path.join(outDir, subDir, imgFile)
            misc.imsave(outPath, frontal[ymin:ymax,xmin:xmax,:])

def getBound(img, shape):
    bleed = 10
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
    xMin -= bleed
    xMax += bleed
    yMin -= 50
    yMax += bleed
    if xMax > len(img[0]):
        xMax = len(img[0])
    if xMin < 0:
        xMin = 0
    if yMax > len(img):
        yMax = len(img)
    if yMin < 0:
        yMin = 0
    return xMin, xMax, yMin, yMax
