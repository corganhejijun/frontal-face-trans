import frontalize
import facial_feature_detector as feature_detection
import camera_calibration as calib
import scipy.io as io
import cv2
import numpy as np
import os
import check_resources as check
from scipy import misc

def frontalize_lfw(dataset, outDir):
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    modelPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "/frontalization_models/model3Ddlib.mat")
    model3D = frontalize.ThreeD_Model(modelPath, 'model_dlib')
    eyemask = np.asarray(io.loadmat('frontalization_models/eyemask.mat')['eyemask'])
    dirList = os.listdir(dataset)
    for index, subDir in dirList:
        print("processing {0}, {1} of total {2}".format(subDir, index, len(dirList)))
        subPath = os.path.join(dataset, subDir)
        for imgFile in os.listdir(subPath):
            imgPath = os.path.join(subPath, imgFile)
            img = cv2.imread(imgPath, 1)
            lmarks = feature_detection.get_landmarks(img)
            proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks[0])
            eyemask = np.asarray(eyemask)
            frontal_raw, frontal_sym = frontalize.frontalize(img, proj_matrix, model3D.ref_U, eyemask)
            if not os.path.exists(os.path.join(outDir, subDir)):
                os.mkdir(os.path.join(outDir, subDir))
            outPath = os.path.join(outDir, subDir, imgFile)
            misc.imsave(outPath, frontal_sym[:, :, ::-1])
