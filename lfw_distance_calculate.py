# -*- coding: utf-8 -*- 
import os
from src.FaceDistance import distance

LFW_DATASET = "datasets\\lfw_aligned"
MODEL_DIR = "models\\20180402-114759"
RESULT_FILE_PATH = "datasets\\lfw_distance.txt"

file = open(RESULT_FILE_PATH, "w")
file.write("name, average, standard, count\n")
file.close()
for dirName in os.listdir(LFW_DATASET):
    print("calculating distance of dir %s" % dirName)
    subDir = os.path.join(LFW_DATASET, dirName)
    avg, std = distance(subDir, MODEL_DIR, 160)
    file = open(RESULT_FILE_PATH, "a")
    file.write("%s, %f, %f, %d\n" % (dirName, avg, std, len(os.listdir(subDir))))
    file.close()