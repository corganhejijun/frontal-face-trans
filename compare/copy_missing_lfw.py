# -*- coding: utf-8 -*-
import os
from shutil import copyfile

LFW_DIR = "../datasets/lfw"
COPY_DIR = "result/lfw_fill"
DEST_DIR = COPY_DIR + "_full"
FILE_EXT = ".png"

if not os.path.exists(DEST_DIR):
    os.mkdir(DEST_DIR)
dirList = os.listdir(LFW_DIR)
for index, dirName in enumerate(dirList):
    print("processing {0}, {1} out of total {2}".format(dirName, index, len(dirList)))
    subDir = os.path.join(LFW_DIR, dirName)
    for file in os.listdir(subDir):
        copyFilePath = os.path.join(COPY_DIR, dirName, file[:-4] + FILE_EXT)
        if not os.path.exists(os.path.join(DEST_DIR, dirName)):
            os.mkdir(os.path.join(DEST_DIR, dirName))
        if not os.path.exists(copyFilePath):
            copyfile(os.path.join(subDir, file), os.path.join(DEST_DIR, dirName, file))
            continue
        destFilePath = os.path.join(DEST_DIR, dirName, file[:-4] + FILE_EXT)
        copyfile(copyFilePath, destFilePath)