# -*- coding: utf-8 -*-
import os
from shutil import copyfile
import cv2
from scipy import misc

LFW_ALIGH = "result/lfw"
OURS_DATA = "result/lfw_fill"
DEST_DIR = "result/lfw_fill_full"

if not os.path.exists(DEST_DIR):
    os.mkdir(DEST_DIR)
dirList = os.listdir(LFW_ALIGH)
for index, dirName in enumerate(dirList):
    print("processing {0}, {1} out of total {2}".format(dirName, index, len(dirList)))
    subDir = os.path.join(LFW_ALIGH, dirName)
    for file in os.listdir(subDir):
        if not os.path.exists(os.path.join(DEST_DIR, dirName)):
            os.mkdir(os.path.join(DEST_DIR, dirName))
        ourfile = os.path.join(OURS_DATA, dirName, file[:-4] + ".png")
        destfile = os.path.join(DEST_DIR, dirName, file)
        if os.path.exists(ourfile):
            img = cv2.cvtColor(cv2.imread(ourfile), cv2.COLOR_BGR2RGB)
            misc.imsave(destfile, img)
        else:
            copyfile(os.path.join(LFW_ALIGH, dirName, file), destfile)