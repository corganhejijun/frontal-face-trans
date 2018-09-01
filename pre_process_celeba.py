# -*- coding: utf-8 -*- 
import os
from shutil import copyfile
from src.util import faceFromDir

CELEBA_IDENTITY_FILE = "datasets\\celeba\\identity_CelebA.txt"
CELEBA_DATASET = "datasets\\celeba\\img_align_celeba"
MIDDLE_DIR = "datasets\\celeba\\celeba_identified"
OUT_DIR = "datasets\\celeba_aligned"

file = open(CELEBA_IDENTITY_FILE, "r")
if not os.path.exists(MIDDLE_DIR):
    os.mkdir(MIDDLE_DIR)
count = 0
for line in file:
    result = line.split(' ')
    fileName = result[0]
    identity = result[1]
    subDir = os.path.join(MIDDLE_DIR, identity[:-1])
    if not os.path.exists(subDir):
        os.mkdir(subDir)
    dest = os.path.join(subDir, fileName)
    if not os.path.exists(dest):
        copyfile(os.path.join(CELEBA_DATASET, fileName), dest)
    count += 1
    if count % 1000 == 0:
        print("%d files had been copied" % count)
file.close()


IN_DIR = "datasets\\celeba\\celeba_identified"
OUT_DIR = "datasets\\celeba_aligned"
SHAPE_MODEL = "models\\shape_predictor_68_face_landmarks.dat"

faceFromDir(IN_DIR, OUT_DIR, SHAPE_MODEL)
